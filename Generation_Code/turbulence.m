function imgI = turbulence(img, F)

% Apply turbulence distortion to an image
% Input: img, distortion field F
% Output: img_turb

sz = size(img);
[X,Y] = meshgrid(1:sz(2), 1:sz(1));
XI = X + F(1:sz(1), 1:sz(2), 1);
YI = Y + F(1:sz(1), 1:sz(2), 2);
if(numel(sz)==2),
    imgI = interp2(double(img), XI, YI);
elseif(numel(sz)==3 && sz(3)==3),
    imgI = img;
    for i=1:3
        imgI(:,:,i) = interp2(double(img(:,:,i)), XI, YI);
    end
end

%figure()
%imshow(imgI, [])

