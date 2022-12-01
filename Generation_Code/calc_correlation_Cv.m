function [C,X,Y] = calc_correlation_Cv(sz, b, f, theta_D)

% Calculates the cross correlation function (matrix)
% AS modified from Marina Alterman

beta_vec = b.beta;
b_lat_vec = b.lat;
b_long_vec = b.long;

% Grid in Physical Coordinates
dx = 1;
[X,Y] = meshgrid(-sz*dx:dx:(sz-1)*dx, -sz*dx:dx:(sz-1)*dx);
r = sqrt(X.^2+Y.^2);
beta_mat = r/(f*theta_D); %assuming small angles. Close to the center of the image. Relative to the middle pixel

% Small correction to avoid exterpolation problems
beta_mat = min(beta_mat, beta_vec(end));

% Interpolate correlation function on grid
b_long_interp = interp1(beta_vec, b_long_vec, beta_mat(:));
b_long_interp_mat = reshape(b_long_interp, size(beta_mat));
b_lat_interp = interp1(beta_vec, b_lat_vec, beta_mat(:));
b_lat_interp_mat = reshape(b_lat_interp, size(beta_mat));

% Covariance Matrix
c11 = b_lat_interp_mat - (b_lat_interp_mat - b_long_interp_mat)./(X.^2+Y.^2).*X.^2;
c21 = -(b_lat_interp_mat - b_long_interp_mat)./(X.^2+Y.^2).*X.*Y;
% c22 = b_lat_interp_mat - (b_lat_interp_mat - b_long_interp_mat)./(X.^2+Y.^2).*Y.^2;
c22 = c11'; % Note that Y = X'
c12 = c21;

% Output
C = cat(4, cat(3, c11, c21), cat(3, c12, c22));
o = sz+1;
C(o, o, :, :) = eye(2);

return


%--------------------------------------------------------------------------
% Check
C11 = C(:,:,1,1);
C22 = C(:,:,2,2);
detC = C(:,:,1,1).*C(:,:,2,2)-C(:,:,1,2).*C(:,:,2,1);
[sum(C11(:) < 0), sum(C22(:) < 0), sum(detC(:) < 0)]

