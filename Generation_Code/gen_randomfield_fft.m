function [F, Fx,Fy] = gen_randomfield_fft(K_gal, n)

sz = [size(K_gal,1) size(K_gal,2)];
if(2.^floor(log2(sz)) ~= sz),
    error('Size must be power of 2')
end
if(sz(1) ~= sz(2)),
    error('Size must be N x N x 2 x 2')
end

% Generate the field with gaussian noise
Fx = zeros([sz/2 n]);
Fy = zeros([sz/2 n]);
h = waitbar(0, 'Generating random fields');
for nn = 1:n, %n - number of instances of random fields
    Zx = randn(sz);
    Zy = randn(sz);
    Zx_gal = fftshift(fftn(Zx));
    Zy_gal = fftshift(fftn(Zy));
    Fx_gal = K_gal(:,:,1,1).*Zx_gal + K_gal(:,:,1,2).*Zy_gal; 
    Fy_gal = K_gal(:,:,2,1).*Zx_gal + K_gal(:,:,2,2).*Zy_gal;
    FFx = ifftn(ifftshift(Fx_gal));
    FFy = ifftn(ifftshift(Fy_gal));
    Fx(:,:,nn) = FFx((sz(1)/4+1):(3*sz(1)/4), (sz(2)/4+1):(3*sz(2)/4));
    Fy(:,:,nn) = FFy((sz(1)/4+1):(3*sz(1)/4), (sz(2)/4+1):(3*sz(2)/4));
    waitbar(nn/n, h)
end
close (h)

varx = sum(sum(K_gal(:,:,1,1).^2 + K_gal(:,:,1,2).^2))/(sz(1)^2);
vary = sum(sum(K_gal(:,:,2,2).^2 + K_gal(:,:,2,1).^2))/(sz(2)^2); % should be the same

Fx =(real(Fx) / sqrt(varx));
Fy = (real(Fy) / sqrt(vary));

F = cat(3, permute(Fx, [1 2 4 3]), permute(Fy, [1 2 4 3]));

return

