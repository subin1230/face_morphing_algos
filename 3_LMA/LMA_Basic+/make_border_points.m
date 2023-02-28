function points = make_border_points(HW,n)
%MAKE_BORDER_POINTS Make a set of points for the border of the image
%   Create points at the corners and n points each side of image with 
%   height HW(1) and width HW(2).
H = HW(1);
W = HW(2);
corner_points = [1,1;W,1;1,H;W,H];
LR_points = zeros(n*2,2);
UD_points = LR_points;
for k = 1:n
    LR_points(2*k-1:2*k,:) = [1,k*(H+1)/(n+1);W,k*(H+1)/(n+1)];
    UD_points(2*k-1:2*k,:) = [k*(W+1)/(n+1),1;k*(W+1)/(n+1),H];
end
points = [corner_points;LR_points;UD_points];
end
