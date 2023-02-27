#!/usr/bin/env python
'''
This code is useing OpenCV to get morphed faces
env: dlib_detect
'''

import numpy as np
import cv2
import sys
import os
import csv
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import gc


# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = [];
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


if __name__ == '__main__' :

    alpha = 0.5

    ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3_frgc/'
    real_path = ro + '2_aligned_1024_bonafide/'  # J_Arcface
    src_point_path = ro + '3_other_morphs/Opencv_landmarks/'
    fake_path = ro + '3_other_morphs/OpenCV/'
    if os.path.exists(fake_path) is False:
        os.makedirs(fake_path)

    subsets = os.listdir(real_path)
    for subset in subsets:
        id_list = os.listdir(real_path + subset)
        for id in id_list:
            imgs = os.listdir(real_path + subset + '/' + id)
            img1 = imgs[0]
            img2 = imgs[1]
            path_img1 = real_path + subset + '/' + id + '/' + img1
            path_img2 = real_path + subset + '/' + id + '/' + img2

            im1 = cv2.imread(path_img1)
            im1 = np.float32(im1)
            points1 = []
            f1 = csv.reader(open(src_point_path + subset + '/' + id + '/' + img1.split('.')[0] + '.csv', 'r'))
            for row in f1:
                points1.append((float(row[0]), float(row[1])))

            im2 = cv2.imread(path_img2)
            im2 = np.float32(im2)
            points2 = []
            f2 = csv.reader(open(src_point_path + subset + '/' + id + '/' + img2.split('.')[0] + '.csv', 'r'))
            for row in f2:
                points2.append((float(row[0]), float(row[1])))

            points = []
            # Compute weighted average point coordinates
            for i in range(0, len(points1)):
                x = (1 - alpha ) * points1[i][0] + alpha * points2[i][0]
                y = (1 - alpha ) * points1[i][1] + alpha * points2[i][1]
                points.append((x,y))

            # Allocate space for final output
            imgMorph = np.zeros(im1.shape, dtype = im1.dtype)

            # Read triangles
            tri = Delaunay(points)
            tri_indx = tri.simplices
            final_list = tri_indx.tolist()
            # print(str(m) + ':' + fold + '_' + name+ '_' + img1.split('.')[0] + '_' + img2.split('.')[0])
            # if fold + '_' + img1.split('.')[0] + '_' + img2.split('.')[0] in skip_set:
            #     print('------------------- skip')
            #     continue
            for line in final_list:
                x = int(line[0])
                y = int(line[1])
                z = int(line[2])
                t1 = [points1[x], points1[y], points1[z]]
                t2 = [points2[x], points2[y], points2[z]]
                t = [ points[x], points[y], points[z]]

                # Morph one triangle at a time.
                morphTriangle(im1, im2, imgMorph, t1, t2, t, alpha)

            # Display Result
            final_name = id
            cv2.imshow("Morphed Face", np.uint8(imgMorph))
            cv2.imwrite(fake_path + final_name + '.jpg', np.uint8(imgMorph))
            # print('d')
