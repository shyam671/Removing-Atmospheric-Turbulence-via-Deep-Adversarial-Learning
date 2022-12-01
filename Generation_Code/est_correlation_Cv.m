function Cest = est_correlation_Cv(F, sz_conv)

sz = size(F);
if (length(sz) == 3), n = 1;
else n = sz(4);
end

ind = (sz(1)/2+1-sz_conv):(sz(1)/2+sz_conv);
[Cxx, Cxy, Cyx, Cyy] = deal(zeros([4*sz_conv-1 4*sz_conv-1 n]));
h = waitbar(0, 'Estimating autocorrelation function');
for nn = 1:n,
    Cxx(:,:,nn) = xcorr2(F(ind,ind,1,nn),F(ind,ind,1,nn));
    Cxy(:,:,nn) = xcorr2(F(ind,ind,1,nn),F(ind,ind,2,nn));
    Cyx(:,:,nn) = Cxy(end:-1:1,end:-1:1,nn);
    Cyy(:,:,nn) = xcorr2(F(ind,ind,2,nn),F(ind,ind,2,nn));
    waitbar(nn/n, h)
end
close (h)

valid = (sz_conv):(3*sz_conv-1);
% N = 4*sz_conv^2;

indexes = [2*sz_conv-2 : 2*sz_conv+2];
middle_calcx = reshape(max(max(Cxx(indexes,indexes,:))),[1,n]);
s2x=mean(middle_calcx);
middle_calcy = reshape(max(max(Cxx(indexes,indexes,:))),[1,n]);
s2y=mean(middle_calcy);

Cxx = mean(Cxx(valid,valid,:), 3) / s2x;
Cxy = mean(Cxy(valid,valid,:), 3) / sqrt(s2x*s2y);
Cyx = mean(Cyx(valid,valid,:), 3) / sqrt(s2x*s2y);
Cyy = mean(Cyy(valid,valid,:), 3) / s2y;

Cest = cat(4, cat(3, Cxx, Cxy), cat(3, Cyx, Cyy));
