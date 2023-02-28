import argparse
import math
import os
import sys
import pickle
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import lpips
from model import Generator
import json
from sklearn.svm import SVC, SVR

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def image_transform(args,file_path):
    resize = min(args.size,256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    img = transform(Image.open(file_path).convert("RGB"))
    imgs.append(img)
    imgs = torch.stack(imgs, 0).to(device)

    return imgs

def image_transform2(args,file_path,size):
    resize = min(args.size,256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    img = transform(Image.open(file_path).convert("RGB").resize((size,size),Image.BILINEAR))
    imgs.append(img)
    imgs = torch.stack(imgs, 0).to(device)

    return imgs


def projection(img_path, args, percept,g_ema,latent_mean,latent_std):

    imgs = image_transform(args, img_path)


    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
    latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    latent_in.requires_grad = True
    optimizer = optim.Adam([latent_in], lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen, _ = g_ema([latent_n], input_is_latent=True)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()

        loss = p_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f};"
                #f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )
    return latent_path[-1]


def morph_two_image(folder, name1, name2, img1, img2, path_img1,path_img2,args, percept,g_ema,latent_mean,latent_std):
    w1 = projection(path_img1,args,percept,g_ema,latent_mean,latent_std)
    # path1 = args.path_to_latent + folder + '/' + name1 + '/'
    # if os.path.exists(path1) is False: os.makedirs(path1)
    # with open(path1 + img1.split('.')[0] + '.npy', 'wb') as f1:
    #     np.save(f1, w1.cpu().numpy())
    w2 = projection(path_img2,args,percept,g_ema,latent_mean,latent_std)
    # path2 = args.path_to_latent + folder + '/' + name2 + '/'
    # if os.path.exists(path2) is False: os.makedirs(path2)
    # with open(path2 + img2.split('.')[0] + '.npy', 'wb') as f2:
    #     np.save(f2, w2.cpu().numpy())

    W = 0.5*w1+0.5*w2
    name = name1 + '_' + img1.split('.')[0] + '+' + name2 + '_' + img2.split('.')[0]
    # path3 = args.path_to_latent + folder + '/'
    # with open(path3 + name + '.npy', 'wb') as f3:
    #     np.save(f3, W.cpu().numpy())

    img_gen, _ = g_ema([W], input_is_latent=True)
    img_ar = make_image(img_gen)
    img_name = name + '.png'
    pil_img = Image.fromarray(img_ar[0])
    dst_p = args.path_to_morph + folder + '/'
    pil_img.save(dst_p + img_name)



# python morphing.py --ckpt  stylegan2-ffhq-config-f.pt --size 1024 --path_to_img1 /path/to/img1/ --path_to_img2 /path/to/img2/

if __name__ == "__main__":
    device = "cuda"
    size=512
    ro = '/home/na/1_Face_morphing/2_data/FRGC-Morphs/frgc/'
    src_path = ro + 'raw_aligned_512_pairs/'
    dst_path_morph = ro + 'raw_aligned_512_pairs_morph_stylegan/'
    if os.path.exists(dst_path_morph) is False:
        os.makedirs(dst_path_morph)
    dst_path_latent = ro + '5_latent_W_aligned_1024/2_items/'
    # if os.path.exists(dst_path_latent) is False:
    #     os.makedirs(dst_path_latent)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='stylegan2-ffhq-config-f.pt')
    parser.add_argument("--path_to_morph", type=str, default=dst_path_morph)
    parser.add_argument("--path_to_latent", type=str, default=dst_path_latent)

    parser.add_argument("--size", type=int, default=size)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=500) # cal W
    parser.add_argument("--noise_regularize", type=float, default=1e5)
    parser.add_argument("--mse", type=float, default=1)
    parser.add_argument("--w_plus", action="store_true")
    args = parser.parse_args()

    n_mean_latent = 10000

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    #latent_avg = torch.from_numpy(np.load(open('/media/ahmed/DiskA1/hftq/dlatent_avg.npy', "rb")))

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    folder_list = os.listdir(src_path)
    for p in range(len(folder_list)):
        folder = folder_list[p]
        print(str(p) + ':' + folder)
        dst_folder_morph = dst_path_morph + folder + '/'
        if os.path.exists(dst_folder_morph) is False:
            os.makedirs(dst_folder_morph)
        # dst_folder_latent = dst_path_latent + folder + '/'
        # if os.path.exists(dst_folder_latent) is False:
        #     os.makedirs(dst_folder_latent)

        name_list = os.listdir(src_path + folder + '/')
        for name in name_list:
            print(name)
            img_list = os.listdir(src_path + folder + '/' + name)
            img1 = img_list[0]
            img2 = img_list[1]
            path_img1 = src_path + folder + '/' + name + '/' + img1
            w1 = projection(path_img1, args, percept, g_ema, latent_mean, latent_std)
            # path1 = args.path_to_latent + folder + '/' + name_list[0] + '/'
            # if os.path.exists(path1) is False: os.makedirs(path1)
            # with open(path1 + img1.split('.')[0] + '.npy', 'wb') as f1:
            #     np.save(f1, w1.cpu().numpy())

            path_img2 = src_path + folder + '/' + name + '/' + img2
            w2 = projection(path_img2, args, percept, g_ema, latent_mean, latent_std)
            # path2 = args.path_to_latent + folder + '/' + name_list[1] + '/'
            # if os.path.exists(path2) is False: os.makedirs(path2)
            # with open(path2 + img2.split('.')[0] + '.npy', 'wb') as f2:
            #     np.save(f2, w2.cpu().numpy())

            W = 0.5 * w1 + 0.5 * w2
            final_name = name + '_' + img1.split('.')[0] + '_' + img2.split('.')[0]
            # path3 = args.path_to_latent + folder + '/'
            # with open(path3 + name + '.npy', 'wb') as f3:
            #     np.save(f3, W.cpu().numpy())

            img_gen, _ = g_ema([W], input_is_latent=True)
            img_ar = make_image(img_gen)
            img_name = final_name + '.png'
            pil_img = Image.fromarray(img_ar[0])
            dst_p = args.path_to_morph + folder + '/'
            pil_img.save(dst_p + img_name)

            # morph_two_image(folder, name_list[0], name_list[1], img1, img2, path1, path2,args,percept,g_ema,latent_mean,latent_std)
