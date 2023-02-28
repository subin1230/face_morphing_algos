function [morph1] = single_morph(img1,img2,facial_landmarks,alpha)
%SINGLE_MORPH Face morph two images onto img2's background
%   Outputs a single morphed image by morphing img1 and img2 onto img2's background
% A basic MATLAB implementation of Neubert et al., “Extended StirTrace
% Benchmarking of Biometric and Forensic Qualities of Morphed Face Images.”
% Combined Morphing Pipeline

% ICAO conform cut (INCOMPLETE may need to include alignment and compression)
    % Add resizing for smaller, and rounding for even
    % Modify:  Cut then align then add boarder points
ICAO_res = [531,413];
ICAO_img1 = img1;%ICAO_cut(img1,ICAO_res);
ICAO_img2 = img2;%ICAO_cut(img2,ICAO_res);

[H,W] = size(ICAO_img1,1:2);

img1_points = facial_landmarks{1};
img2_points = facial_landmarks{2};

num_edgePoints = 4;

% Make keypoints along the border of the image
border_points = make_border_points([H,W],num_edgePoints);

img1_points = [border_points;img1_points];
img2_points = [border_points;img2_points];

% Find and remove duplicate landmarks (may need to remove lmks 62-64)
[a1,b1,c1] = unique(img1_points,'rows','stable');
if size(a1,1) < size(c1,1)
    img1_points = a1;
    img2_points = img2_points(b1,:);
    disp("Duplicate Points Detected in img1")
end
[a2,b2,c2] = unique(img2_points,'rows','stable');
if size(a2,1) < size(c2,1)
    img2_points = a2;
    img1_points = img1_points(b2,:);
    disp("Duplicate Points Detected in img2")
end

% lmk_img1 = show_landmarks(0.85*ones(size(img1)+[20,20,0]),img1_points+[10,10],'blue',10);
% lmk_img2 = show_landmarks(0.85*ones(size(img2)+[20,20,0]),img2_points+[10,10],'blue',10);

% Blend Geometry
avg_points = alpha.*img1_points + (1-alpha).*img2_points;

% Triangulartion
DT = delaunayTriangulation(avg_points);
triangles = DT.ConnectivityList;

% Warping (NEEDS CLEANING)
%    Initialize warped images
img1_warped = zeros(size(ICAO_img1));
img2_warped = zeros(size(ICAO_img2));

%    Calculate the transform matrix for each triangle pair
img1_transforms = zeros(3,3,size(triangles,1));
img2_transforms = zeros(3,3,size(triangles,1));

for n = 1:size(triangles,1)
    avg_tri = avg_points(triangles(n,:),:);
    img1_tri = img1_points(triangles(n,:),:);
    img2_tri = img2_points(triangles(n,:),:);
    img1_tform = fitgeotrans(img1_tri,avg_tri,'affine');
    img2_tform = fitgeotrans(img2_tri,avg_tri,'affine');
    img1_transforms(:,:,n) = img1_tform.T;
    img2_transforms(:,:,n) = img2_tform.T;
    
    %     Find I_i, the set of pixels enclosed in triangle i
    avg_mask = poly2mask(avg_tri(:,1),avg_tri(:,2),size(ICAO_img2,1),size(ICAO_img2,2));
    
    %     Apply Mapping functions such that Isw_i = fs_i(Is_i)
    img1_warp_bit = imwarp(ICAO_img1,img1_tform,'OutputView',imref2d(size(ICAO_img1)));
    img2_warp_bit = imwarp(ICAO_img2,img2_tform,'OutputView',imref2d(size(ICAO_img1)));
    img1_warped = img1_warped + double(img1_warp_bit).*double(avg_mask);
    img2_warped = img2_warped + double(img2_warp_bit).*double(avg_mask);

end

% Cross Disolve (average)
blended_img = alpha*img1_warped+(1-alpha)*img2_warped;

% Cut Convex hull
CDT = delaunayTriangulation(avg_points((5+4*num_edgePoints):end,:));
C = convexHull(CDT);
x = CDT.Points(C,1);
y = CDT.Points(C,2);
conv_mask = double(poly2mask(x,y,H,W));

CDT = delaunayTriangulation(img2_points((5+4*num_edgePoints):end,:));
C = convexHull(CDT);
x = CDT.Points(C,1);
y = CDT.Points(C,2);
img2_mask = double(poly2mask(x,y,H,W));


% Insert Blended face
% morph1 = ICAO_img1 .* (1-conv_mask) + avg_face;

% Poisson Image Blending
img2f = imGradFeature(ICAO_img2);
Bf = imGradFeature(blended_img);

% Shrink splicing mask to avoid including background pixels
img2_splice_mask = conv_mask.*img2_mask;

for i = 1:H
    for j = 1:W
        if img2_splice_mask(i,j) ~= 0
            img2f(i,j,:,:) = Bf(i,j,:,:);
        end
    end
end

% Splicing onto img2
param = buildModPoissonParam( size(img2f) );
morph1 = modPoisson( img2f, param);

end