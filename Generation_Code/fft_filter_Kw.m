function [C_fft, K_fft] = fft_filter_Kw(C,X,Y)

% Compute power spectrum using FFT

sz = size(C,1);
if(2.^floor(log2(sz)) ~= sz),
    error('Size must be power of 2')
end

% Multiply by Gaussian window
d = sz(1)/2/4; % standard deviation of Gaussian window
r2 = X.^2 + Y.^2;
g = exp(-r2/2/d^2);
%------
% Multiply by Blackman window
w=blackman(2048);
w_2D = w(:)*w(:).';
%-----------

C11 = C(:,:,1,1).* w_2D;
C21 = C(:,:,2,1).* w_2D;

% Fourier Transform (Origin Shifted to Node (1,1))
C11_fft = max(real(fftshift(fft2(fftshift(C11)))), 0);
C21_fft = real(fftshift(fft2(fftshift(C21))));
C12_fft = C21_fft; % Note that C12 = C21
C22_fft = C11_fft'; % Note that C22 = C11'

C_fft = cat(4, cat(3, C11_fft, C21_fft), cat(3, C12_fft, C22_fft));


%--------------------------------------------------------------------------
% Compute filter frequency response

% Use explicit square root formula for 2x2 matrices (from Wikipedia)
detC_fft = abs(C_fft(:,:,1,1).*C_fft(:,:,2,2) - C_fft(:,:,1,2).*C_fft(:,:,2,1));
s = sqrt(detC_fft);
t = sqrt(C11_fft + C22_fft + 2*s);

K11_fft = (C11_fft + s) ./ t;
K21_fft = C21_fft ./ t;
K12_fft = K21_fft;
K22_fft = K11_fft';

K_fft = cat(4, cat(3, K11_fft, K21_fft), cat(3, K12_fft, K22_fft));

% Numerical
% K_fft = C_fft;
% for i = 1:sz(1),
%     for j = i:sz(2),
%         K_fft(i,j,:,:) = sqrtm(squeeze(C_fft(i,j,:,:)));
%     end
% end

return


%--------------------------------------------------------------------------
% Check positive definiteness
C11_fft = C_fft(:,:,1,1);
C22_fft = C_fft(:,:,2,2);
[sum(C11_fft(:) < 0), sum(C22_fft(:) < 0), sum(detC_fft(:) < 0)]

K11 = K_fft(:,:,1,1);
K22 = K_fft(:,:,2,2);
detK_fft = K_fft(:,:,1,1).*K_fft(:,:,2,2)-K_fft(:,:,1,2).*K_fft(:,:,2,1);
[sum(K11(:) < 0), sum(K22(:) < 0), sum(detK_fft(:) < 0)]

% Check determinant
figure;
imagesc(detC_fft)
title('det(C\_fft)'), axis square, colorbar
