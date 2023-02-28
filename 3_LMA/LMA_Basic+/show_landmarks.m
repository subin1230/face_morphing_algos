function [im] = show_landmarks(im,landmarks,color,rad)
%SHOW_LANDMARKS Show the landmarks on an image

if ~exist('rad','var')
    rad = 1;
end
if exist('color','var')
    for k = 1:size(landmarks,1)
        im = insertShape(im, 'FilledCircle', [landmarks(k,:),rad],'Color',color);
    end
else
    for k = 1:size(landmarks,1)
        im = insertShape(im, 'FilledCircle', [landmarks(k,:),rad]);
        
    end
end

