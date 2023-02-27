# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:38:45 2021
@author: Jake
env: Facemorpher
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

import os
import locator
import aligner
import warper
import blender
import plotter
import videoer

def offset_image(moveable_img, moveable_points, static_img, static_points):
    offset = calc_offset(moveable_points,static_points)
    new_img = np.roll(moveable_img,offset[1],axis=0)
    new_img = np.roll(new_img,offset[0],axis=1)
    new_points = np.add(moveable_points,offset)
    return(new_img, new_points)

def inv_offset_image(moveable_img, moveable_points, offset):
    new_img = np.roll(moveable_img,-1*offset[0],axis=0)
    new_img = np.roll(new_img,-1*offset[1],axis=1)
    new_points = np.add(moveable_points,-1*offset)
    return(new_img, new_points)

def calc_offset(points1,points2):
    offset = np.subtract(points2[33],points1[33])
    return offset

def list_imgpaths(images_folder=None, src_image=None, dest_image=None):
  if images_folder is None:
    yield src_image
    yield dest_image
  else:
    for fname in os.listdir(images_folder):
      if (fname.lower().endswith('.jpg') or
         fname.lower().endswith('.png') or
         fname.lower().endswith('.jpeg')):
        yield os.path.join(images_folder, fname)
        
def morph_images(src, dest, size, percent):
    orig_src_img = cv2.imread(src)
    orig_src_img = cv2.cvtColor(orig_src_img, cv2.COLOR_BGR2RGB)
    src_points = locator.face_points(orig_src_img)

    dest_img = cv2.imread(dest)
    dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2RGB)
    dest_points = locator.face_points(dest_img)
    
    # Calculate the offset of the face centers and adjust
    offset = calc_offset(src_points,dest_points)
    src_img, src_points = offset_image(orig_src_img, src_points, dest_img, dest_points)
    
    # Average Landmarks
    avg_points = locator.weighted_average_points(src_points, dest_points, percent)
    
    # Perform transform & cut
    src_face_warped = warper.warp_image(src_img, src_points, avg_points, size)
    dest_face_warped = warper.warp_image(dest_img, dest_points, avg_points, size)
    
    # Blend 
    avg_face = blender.weighted_average(src_face_warped, dest_face_warped, percent)
    mask = blender.mask_from_points(avg_face.shape[:2], avg_points)
    
    # plt.subplot(1, 3, 1)
    # plt.imshow(src_img)
    # plt.title("Src")
    # plt.subplot(1, 3, 2)
    # plt.imshow(dest_img)
    # plt.title("Dest")
    # plt.subplot(1, 3, 3)
    # plt.imshow(avg_face)
    # plt.title("Average")
    # plt.show()
    
    # Splicing onto src
    src_mask = blender.mask_from_points(src_img.shape[:2], src_points)
    
    # Splicing onto dest
    dest_mask = blender.mask_from_points(dest_img.shape[:2], dest_points)
    
    output_img1 = blender.poisson_blend(avg_face,orig_src_img,np.multiply(src_mask,dest_mask),np.flip(-1*offset))
    # plt.imshow(output_img1)
    # plt.title("PB AVG w/ src&dest mask")
    # plt.show()
    
    output_img2 = blender.poisson_blend(avg_face,dest_img,np.multiply(src_mask,mask))
    # plt.imshow(output_img2)
    # plt.title("PB AVG w/ src&dest mask")
    # plt.show()
    return(output_img1, output_img2)


# main
ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3_frgc/'
src_path = ro + '2_aligned_1024_bonafide/'  # J_Arcface
dst_path = ro + '3_other_morphs/FaceMorpher/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

percent = 0.5
size = (1024,1024)

subsets = os.listdir(src_path)
for subset in subsets:
    id_list = os.listdir(src_path + subset)
    for id in id_list:
        imgs = os.listdir(src_path + subset + '/' + id)
        img1 = imgs[0]
        img2 = imgs[1]
        path_img1 = src_path + subset + '/' + id + '/' + img1
        path_img2 = src_path + subset + '/' + id + '/' + img2

        o1, o2 = morph_images(path_img1, path_img2, size, percent)

        final_name = id
        dst_img = dst_path + final_name + '.jpg'

        cv2.imwrite(dst_img, cv2.cvtColor(o1, cv2.COLOR_RGB2BGR))  # src in background

