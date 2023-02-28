% Generate Dataset of morphs from images located in 'input_path'
% Last Update: 1/24/2021

clear
clc
close all

input_path = '.\test\';
dataset_name = 'test';

output_path = ['.\',dataset_name,'_Results\'];
mkdir(output_path)

% Initialize
imds = imageDatastore(input_path);
number_of_morphs = 2*nchoosek(size(imds.Files,1),2);
varNames = {'filename','srcImg1','srcImg2(background)','dataset'};
metadata = table('Size',[number_of_morphs,4],'VariableTypes',{'string','string','string','string'},'VariableNames',varNames);
counter = 0;

% Import Landmark Extractor
addpath(genpath('..\find_face_landmarks-1.2-x64-vc14-release'))
modelFile = '..\shape_predictor_68_face_landmarks.dat';

fprintf("Completed morphs: %6.0f of %6.0f.\n",counter,number_of_morphs)

for n1 = 1:size(imds.Files,1)
    [~,name1,~] = fileparts(imds.Files{n1});
    
    img1 = double(readimage(imds,n1));
    img1_points = double(find_face_landmarks(modelFile, uint8(img1)).faces(1).landmarks);
    img1_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks
    
    % Estimate iris locations for image 1
    L_eye1 = mean(img1_points(37:42,:));
    R_eye1 = mean(img1_points(43:48,:));
    
    for n2 = n1+1:size(imds.Files,1)
        [~,name2,~] = fileparts(imds.Files{n2});
        
        fn1 = [name1,'+',name2,'.jpg'];
        fn2 = [name2,'+',name1,'.jpg'];
        
        metadata(counter+1,:) = {fn1,name1,name2,dataset_name};
        metadata(counter+2,:) = {fn2,name2,name1,dataset_name};
        
        img2 = double(readimage(imds,n2));
        img2_points = double(find_face_landmarks(modelFile, uint8(img2)).faces(1).landmarks);
        img2_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks
        
        % Estimate iris locations for image 2
        L_eye2 = mean(img2_points(37:42,:));
        R_eye2 = mean(img2_points(43:48,:));
        
        lmk_img1 = show_landmarks(0.85*ones(size(img1)),img1_points,'blue',3);
        lmk_img1 = show_landmarks(lmk_img1,[L_eye1;R_eye1],'red',3);
        figure()
        imshow(lmk_img1)
        lmk_img2 = show_landmarks(0.85*ones(size(img2)),img2_points,'blue',10);
        lmk_img2 = show_landmarks(lmk_img2,[L_eye2;R_eye2],'red',10);
        figure()
        imshow(lmk_img2)
        
        % Align images according to iris loactions
        [aligned_img1] = align_face(img1,img2,{[L_eye1;R_eye1];[L_eye2;R_eye2]},true);
        [aligned_img2] = align_face(img2,img1,{[L_eye2;R_eye2];[L_eye1;R_eye1]},true);
        
        alignedImg1_points = double(find_face_landmarks(modelFile, uint8(aligned_img1)).faces(1).landmarks);
        alignedImg1_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks
        alignedImg2_points = double(find_face_landmarks(modelFile, uint8(aligned_img2)).faces(1).landmarks);
        alignedImg2_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks
        
        figure()
        imshow(aligned_img1/255)
        figure()
        imshow(aligned_img2/255)
        
        % Generate morph aligned to image 2
        facial_landmarks{1} = alignedImg1_points;
        facial_landmarks{2} = img2_points;
        [M2] = single_morph(aligned_img1,img2,facial_landmarks,0.5);
        
        % Generate morph aligned to image 1
        facial_landmarks{1} = alignedImg2_points;
        facial_landmarks{2} = img1_points;
        [M1] = single_morph(aligned_img2,img1,facial_landmarks,0.5);
        
        imwrite(uint8(M1),[output_path,fn1])
        imwrite(uint8(M2),[output_path,fn2])
        counter = counter + 2;
        
        if mod(counter,100) <= 1 
            fprintf("Completed morphs: %6.0f of %6.0f.\n",counter,number_of_morphs)
        end
    end  
end
writetable(sortrows(metadata),[output_path,'metadata.csv'])