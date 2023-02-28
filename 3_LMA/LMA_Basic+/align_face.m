function [img1rss] = align_face(img1,img2,eye_landmarks,scaling)
%ALIGN_FACE Align faces based on respective landmarks
%   Aligns faces by rotating, shifting and scaling img1 until the eyes of
%   both imgs are overlayed.
% Inputs:
%   img1 - image to align to img2
%   img2 - image to align img1 to
%   eye_landmarks - cell array containing the iris landmarks of img1 & img2
%       in that order.  Each cell contains a 2x2 array with the rows
%       corresponding to different eyes.
%   scaling - set to true to scale img1 to img2
% Outputs:
%   img1rss - img1 rotated scaled and shifted to align the irises in the
%   two images.


img1_points = eye_landmarks{1};
img2_points = eye_landmarks{2};

L_eye1 = img1_points(1,:);
L_eye2 = img2_points(1,:);
R_eye1 = img1_points(2,:);
R_eye2 = img2_points(2,:);

% Rotate
x = double(R_eye1(1) - L_eye1(1));
y = double(-1*R_eye1(2) + L_eye1(2));
theta1 = (180/pi)*atan2(y,x);

x = double(R_eye2(1) - L_eye2(1));
y = double(-1*R_eye2(2) + L_eye2(2));
theta2 = (180/pi)*atan2(y,x);

img1r = imrotate(img1,(theta2 - theta1),'nearest','crop');

[H1,W1] = size(img1,1:2);
C1 = [W1/2,H1/2];

rot_points = rotate_coordinates([L_eye1;R_eye1],(theta2 - theta1),C1);
L_eye1 = rot_points(1,:);
R_eye1 = rot_points(2,:);

% Scale (make distance between eyes the same)
if scaling
    D1 = sqrt(sum((L_eye1 - R_eye1).^2));
    D2 = sqrt(sum((L_eye2 - R_eye2).^2));
    
    % Scale img1
    img1rs = imresize(img1r,D2/D1,'bicubic');
        
    % Remap Eye Landmarks onto img1
    L_eye1 = L_eye1*D2/D1;
    R_eye1 = R_eye1*D2/D1;
    
else
    img1rs = img1r;
end

[H1,W1] = size(img1rs,1:2);
[H2,W2] = size(img2,1:2);

if H2 > H1 || W2 > W1 % img1 is smaller than img2
    % Pad img1 to the size of img2
    img1rs = padarray(img1rs,[H2-H1,W2-W1],0,'post');
    [H1,W1] = size(img1rs,1:2);
end

% Shift (place eyes on top of eachother)
% Shift Img1
D = round(mean([L_eye2;R_eye2] - [L_eye1;R_eye1],1));
L_eye1 = L_eye1 + D;
R_eye1 = R_eye1 + D;
img1rss = circshift(img1rs,D(2),1);
img1rss = circshift(img1rss,D(1),2);

% Crop img1
if H1 > H2 || W1 > W2 % img1 is larger than img2
    img1rss = img1rss(1:H2,1:W2,:);
end

end

