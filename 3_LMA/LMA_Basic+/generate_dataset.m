% Generate Dataset of morphs from images located in 'input_path'
% Last Update: 5/4/2021

% clear
% clc
% close all

% Import Landmark Extractor
ro = 'D:\2_zn_research\1_Morphing_generation_single\1_code\0_Jake_code\Morphing_Code\MATLAB Implementations\LMA_Basic+\'
addpath(genpath(['C:\find_face_landmarks']))
addpath(genpath(['C:\find_face_landmarks\interfaces\matlab'])) 

modelFile = [ro 'shape_predictor_68_face_landmarks.dat'];


input_path = 'D:\2_zn_research\1_Morphing_generation_single\2_data\3_self\1_real_faces\';
dataset_name = 'self';

output_path = 'D:\2_zn_research\1_Morphing_generation_single\2_data\3_self\2_morph_LMA\';
% mkdir(output_path)

d       = dir(fullfile(input_path));
dirlist = d([d.isdir]);
dirlist = dirlist(~ismember({dirlist.name}, {'.','..'}));
% a  =size(dirlist,1)
for i=1:size(dirlist,1)
    folder = dirlist(i).name  
    
    if ~exist([output_path folder '\'], 'dir')
        mkdir([output_path folder '\'])
    end
    
    d2       = dir(fullfile([input_path folder '\']));
    dirlist2 = d2([d2.isdir]);
    dirlist2 = dirlist2(~ismember({dirlist2.name}, {'.','..'}));

    Images1 = dir([input_path folder '\' dirlist2(1).name, '\*.png']);
    len1 = size(Images1,1)
    
    Images2 = dir([input_path folder '\' dirlist2(2).name, '\*.png']);
    len2 = size(Images2,1)
    
    for m=1:len1
        img_name1 = Images1(m).name
        imx1 = [input_path folder '\' dirlist2(1).name, '\' img_name1]   
        img1 = double(imread(imx1));
        img1_points = double(find_face_landmarks(modelFile, uint8(img1)).faces(1).landmarks);
        img1_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks

        % Estimate iris locations for image 1
        L_eye1 = mean(img1_points(37:42,:));
        R_eye1 = mean(img1_points(43:48,:));
    
    
        for n=1:len2
            img_name2 = Images2(m).name
            imx2 = [input_path folder '\' dirlist2(2).name, '\' img_name2]          
            img2 = double(imread(imx2));
            img2_points = double(find_face_landmarks(modelFile, uint8(img2)).faces(1).landmarks);
            img2_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks

            % Estimate iris locations for image 2
            L_eye2 = mean(img2_points(37:42,:));
            R_eye2 = mean(img2_points(43:48,:));
            
            name1 = substr(img_name1, 1, 3)
            name2 = substr(img_name2, 1, 3)            
            fn1 = [dirlist2(1).name '_' name1 , '+', dirlist2(2).name, '_', name2,'.jpg']; %.jpg as in the AMSL and londondb dataset
            fn2 = [dirlist2(2).name '_' name2,'+', dirlist2(1).name, '_', name1,'.jpg'];
        
            % Align images according to iris loactions
            [aligned_img1] = align_face(img1,img2,{[L_eye1;R_eye1];[L_eye2;R_eye2]},true);
            [aligned_img2] = align_face(img2,img1,{[L_eye2;R_eye2];[L_eye1;R_eye1]},true);

            alignedImg1_points = double(find_face_landmarks(modelFile, uint8(aligned_img1)).faces(1).landmarks);
            alignedImg1_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks
            alignedImg2_points = double(find_face_landmarks(modelFile, uint8(aligned_img2)).faces(1).landmarks);
            alignedImg2_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks

            % Generate morph aligned to image 2
            facial_landmarks{1} = alignedImg1_points;
            facial_landmarks{2} = img2_points;
            [M2] = single_morph(aligned_img1,img2,facial_landmarks,0.5);

            % Generate morph aligned to image 1
            facial_landmarks{1} = alignedImg2_points;
            facial_landmarks{2} = img1_points;
            [M1] = single_morph(aligned_img2,img1,facial_landmarks,0.5);
        
            % Save Images
            imwrite(uint8(M1),[output_path folder '\',fn1])
            imwrite(uint8(M2),[output_path folder '\',fn2])
            
        end
    end       
end
    

% D:\2_zn_research\1_Morphing_generation_single\1_code\0_Jake_code\Morphing_Code\MATLAB Implementations\LMA_Basic+\find_face_landmarks-1.2-x64-vc14-release\bin;
% C:\Program Files\CMake\bin;
% C:\Users\Nana\AppData\Roaming\cabal\bin;
% C:\Users\Nana\AppData\Roaming\local\bin;
% C:\Program Files\Intel\WiFi\bin\;
% C:\Program Files\Common Files\Intel\WirelessCommon\;%PyCharm Community Edition%


% Initialize
imds = imageDatastore(input_path);
number_of_morphs = 2*nchoosek(size(imds.Files,1),2);
varNames = {'filename','srcImg1','srcImg2(background)','dataset'};
metadata = table('Size',[number_of_morphs,4],'VariableTypes',{'string','string','string','string'},'VariableNames',varNames);
counter = 0;



% fprintf("Completed morphs: %6.0f of %6.0f.\n",counter,number_of_morphs)

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
        
        fn1 = [name1,'+',name2,'.jpg']; %.jpg as in the AMSL and londondb dataset
        fn2 = [name2,'+',name1,'.jpg'];
        
        metadata(counter+1,:) = {fn1,name1,name2,dataset_name};
        metadata(counter+2,:) = {fn2,name2,name1,dataset_name};
        
        img2 = double(readimage(imds,n2));
        img2_points = double(find_face_landmarks(modelFile, uint8(img2)).faces(1).landmarks);
        img2_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks
        
        % Estimate iris locations for image 2
        L_eye2 = mean(img2_points(37:42,:));
        R_eye2 = mean(img2_points(43:48,:));
        
        % Align images according to iris loactions
        [aligned_img1] = align_face(img1,img2,{[L_eye1;R_eye1];[L_eye2;R_eye2]},true);
        [aligned_img2] = align_face(img2,img1,{[L_eye2;R_eye2];[L_eye1;R_eye1]},true);
        
        alignedImg1_points = double(find_face_landmarks(modelFile, uint8(aligned_img1)).faces(1).landmarks);
        alignedImg1_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks
        alignedImg2_points = double(find_face_landmarks(modelFile, uint8(aligned_img2)).faces(1).landmarks);
        alignedImg2_points(62:64,:) = []; %Throw out bottom contour of upper lip landmarks
        
        % Generate morph aligned to image 2
        facial_landmarks{1} = alignedImg1_points;
        facial_landmarks{2} = img2_points;
        [M2] = single_morph(aligned_img1,img2,facial_landmarks,0.5);
        
        % Generate morph aligned to image 1
        facial_landmarks{1} = alignedImg2_points;
        facial_landmarks{2} = img1_points;
        [M1] = single_morph(aligned_img2,img1,facial_landmarks,0.5);
        
        % Save Images
        imwrite(uint8(M1),[output_path,fn1])
        imwrite(uint8(M2),[output_path,fn2])
        counter = counter + 2;
        
        if mod(counter,100) <= 1 
%             fprintf("Completed morphs: %6.0f of %6.0f.\n",counter,number_of_morphs)
        end
    end  
end

% Save Labels
writetable(sortrows(metadata),[output_path,'metadata.csv'])






