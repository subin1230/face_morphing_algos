clear
clc
close all

% Initialize for TEST
path = '.\test\';
imds = imageDatastore(path);
number_of_sets = nchoosek(size(imds.Files,1),2);

% Extract Landmarks
addpath(genpath('..\find_face_landmarks-1.2-x64-vc14-release'))
modelFile = '..\shape_predictor_68_face_landmarks.dat';

for n1 = 1:size(imds.Files,1)
    for n2 = n1+1:size(imds.Files,1)
        fn1 = sprintf('%1.0f+%1.0f.jpg',n1,n2);
        fn2 = sprintf('%1.0f+%1.0f.jpg',n2,n1);
        
        img1 = double(readimage(imds,n1));
        img2 = double(readimage(imds,n2));
        
        img1_points = double(find_face_landmarks(modelFile, uint8(img1)).faces.landmarks);
        img1_points(62:64,:) = [];
        img2_points = double(find_face_landmarks(modelFile, uint8(img2)).faces.landmarks);
        img2_points(62:64,:) = [];
        
        L_eye1 = mean(img1_points(37:42,:));
        L_eye2 = mean(img2_points(37:42,:));
        R_eye1 = mean(img1_points(43:48,:));
        R_eye2 = mean(img2_points(43:48,:));
        
        [aligned_img1] = align_face(img1,img2,{[L_eye1;R_eye1];[L_eye2;R_eye2]},true);
        [aligned_img2] = align_face(img2,img1,{[L_eye2;R_eye2];[L_eye1;R_eye1]},true);
        
        a1_points = double(find_face_landmarks(modelFile, uint8(aligned_img1)).faces.landmarks);
        a1_points(62:64,:) = [];
        a2_points = double(find_face_landmarks(modelFile, uint8(aligned_img2)).faces.landmarks);
        a2_points(62:64,:) = [];
        
        facial_landmarks{1} = a1_points;
        facial_landmarks{2} = img2_points;
        [M2] = single_morph(aligned_img1,img2,facial_landmarks,0.5);
        
        facial_landmarks{1} = a2_points;
        facial_landmarks{2} = img1_points;
        [M1] = single_morph(aligned_img2,img1,facial_landmarks,0.5);

        imwrite(uint8(M1),['.\align_face_test\',fn1])
        imwrite(uint8(M2),['.\align_face_test\',fn2])
    end
end