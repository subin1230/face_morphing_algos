function [XY] = rotate_coordinates(XY,theta,C)
%ROTATE_COORDINATES Rotate [x,y] coordinates by theta with center [cx,cy]

R = [cosd(theta),-sind(theta);sind(theta),cosd(theta)];
s = XY - C;
XY = s*R;
XY = XY + C;
end

