function [K, KK] = fft_kernel(K_gal,X,Y,sz_conv)

% Compute impulse response using FFT

sz = size(K_gal,1);
if(2.^floor(log2(sz)) ~= sz),
    error('Size must be power of 2')
end

% Multiply by Gaussian window
d = sz/2/4; % standard deviation of Gaussian window
r2 = X.^2 + Y.^2;
g = exp(-r2/2/d^2);

% Multiply by Blackman window
w=blackman(sz);
w_2D = w(:)*w(:).';

K11_gal = K_gal(:,:,1,1) .* w_2D;
K21_gal = K_gal(:,:,2,1) .* w_2D;

% Fourier Transform (Origin Shifted to Node (1,1))
K11 = real(ifftshift(ifftn(ifftshift(K11_gal))));
K21 = real(ifftshift(ifftn(ifftshift(K21_gal))));
K12 = K21;  % Note that K12_gal = K21_gal
K22 = K11'; % Note that K22_gal = K11_gal'

% Assign output
K = cat(4, cat(3, K11, K21), cat(3, K12, K22));

% Check autoconvolution of impulse response
ind = (sz/2+1-sz_conv):(sz/2+sz_conv);
KK = zeros(size(K));
KK(ind,ind,1,1) = convn(K11(ind,ind), K11(ind,ind), 'same') + ...
    convn(K12(ind,ind), K21(ind,ind), 'same');
KK(ind,ind,1,2) = convn(K11(ind,ind), K12(ind,ind), 'same') + ...
    convn(K12(ind,ind), K22(ind,ind), 'same');
KK(ind,ind,2,1) = KK(ind,ind,1,2);
KK(ind,ind,2,2) = KK(ind,ind,1,1)';

return

%--------------------------------------------------------------------------
% Check positive definiteness
detC_fft = K11.*K22-K12.*K21;
[sum(K11(:) < 0), sum(K22(:) < 0), sum(detC_fft(:) < 0)]

