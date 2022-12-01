% % Generate turbulence random field
% % AS modified from Marina Alterman
% % clear, clc
% % close all
% 
% % Imaging parameters
% fm = 300; %focal distance in mm
% D = fm/5.6*10^(-3); %lens diameter (6 to 8 cm)
% pixel_size = 4e-3; %in mm
% f = fm/pixel_size; %focal distance in pixels
% 
% L = 2e3; %distance in meters (2 to 10 km)
% h = 4; %height in meters (2 to 500 m)
% L0 = 0.4*h; %L0 outer scale
% theta_D = D/L; %angular size of images at distance L
% gamma = L0/D; %AA correlation parameter
% lamda = 550e-9;
% 
% sz = 1024; % image size in pixels (1024x1024 to 4096x4096)
% sz_conv = sz/4; % size in pixels for convolution testing
% % sz_conv should be equal to sz but it takes too long to compute
% 
% % wave = 'plane';
% wave = 'sphere';
% Cn_2 = 1e-12; %typical for daytime in June in meters^(-2/3); (Not needed)
% sigma2_A = (AAvariance(Cn_2, D, L, lamda, wave, 'full')); % AA standard deviation in radians 
% sigma_A=sqrt(sigma2_A);
% sigma_r = f*sigma_A;
% 
% %% Compute AA correlation
% tic
% b = AAcorrelation(gamma);
% toc
% save(['AA_gamma', num2str(round(gamma))], 'b')
% 
% %% Vector autocorrelation function
% [C,X,Y] = calc_correlation_Cv(sz, b, f, theta_D);
% 
% %figure(2);
% %subplot(221), imagesc(X(1,:),Y(:,1),C(:,:,1,1), [0 1])
% %title('C11'), axis xy square; colorbar
% %subplot(222), imagesc(X(1,:),Y(:,1),C(:,:,2,1), 0.1*[-1 1])
% %title('C21'), axis xy square; colorbar
% %subplot(223), imagesc(X(1,:),Y(:,1),C(:,:,1,2), 0.1*[-1 1])
% %title('C12'), axis xy square; colorbar
% %subplot(224), imagesc(X(1,:),Y(:,1),C(:,:,2,2), [0 1])
% %title('C22'), axis xy square; colorbar
% 
% ind = (sz+1-sz_conv):(sz+sz_conv);
% %figure(3);
% %subplot(221), imagesc(X(1,ind),Y(ind,1),C(ind,ind,1,1), [0 1])
% %title('C11'), axis xy square; colorbar
% %subplot(222), imagesc(X(1,ind),Y(ind,1),C(ind,ind,2,1), 0.1*[-1 1])
% %title('C21'), axis xy square; colorbar
% %subplot(223), imagesc(X(1,ind),Y(ind,1),C(ind,ind,1,2), 0.1*[-1 1])
% %title('C12'), axis xy square; colorbar
% %subplot(224), imagesc(X(1,ind),Y(ind,1),C(ind,ind,2,2), [0 1])
% %title('C22'), axis xy square; colorbar
% 
% 
% %% Compute filter spectrum via FFT
% [C_fft, K_fft] = fft_filter_Kw(C,X,Y);
% 
% ind = (sz+1-sz_conv):(sz+sz_conv);
% %figure(4);
% %subplot(221), imagesc(X(1,ind),Y(ind,1),C_fft(ind,ind,1,1))
% %title('C\_fft 11'), axis xy square; colorbar
% %subplot(222), imagesc(X(1,ind),Y(ind,1),C_fft(ind,ind,2,1))
% %title('C\_fft 21'), axis xy square; colorbar
% %subplot(223), imagesc(X(1,ind),Y(ind,1),C_fft(ind,ind,1,2))
% %title('C\_fft 12'), axis xy square; colorbar
% %subplot(224), imagesc(X(1,ind),Y(ind,1),C_fft(ind,ind,2,2))
% %title('C\_fft 22'), axis xy square; colorbar
% 
% %figure(5);
% %subplot(221), imagesc(X(1,ind),Y(ind,1),K_fft(ind,ind,1,1))
% %title('K\_fft 11'), axis xy square; colorbar
% %subplot(222), imagesc(X(1,ind),Y(ind,1),K_fft(ind,ind,2,1))
% %title('K\_fft 21'), axis xy square; colorbar
% %subplot(223), imagesc(X(1,ind),Y(ind,1),K_fft(ind,ind,1,2))
% %title('K\_fft 12'), axis xy square; colorbar
% %subplot(224), imagesc(X(1,ind),Y(ind,1),K_fft(ind,ind,2,2))
% %title('K\_fft 22'), axis xy square; colorbar
% 
% 
% %% Impulse response via FFT
% tic
% [K, KK] = fft_kernel(K_fft,X,Y,sz_conv);
% toc
% 
% ind = (sz+1-sz_conv):(sz+sz_conv);
% %figure(6);
% %subplot(221), imagesc(X(1,ind),Y(ind,1),K(ind,ind,1,1))
% %title('K 11'), axis xy square; colorbar
% %subplot(222), imagesc(X(1,ind),Y(ind,1),K(ind,ind,2,1))
% %title('K 21'), axis xy square; colorbar
% %subplot(223), imagesc(X(1,ind),Y(ind,1),K(ind,ind,1,2))
% %title('K 12'), axis xy square; colorbar
% %subplot(224), imagesc(X(1,ind),Y(ind,1),K(ind,ind,2,2))
% %title('K 22'), axis xy square; colorbar
% 
% %figure(7);
% %subplot(221), imagesc(X(1,ind),Y(ind,1),KK(ind,ind,1,1), [0 1])
% %title('KK 11'), axis xy square; colorbar
% %subplot(222), imagesc(X(1,ind),Y(ind,1),KK(ind,ind,2,1), 0.1*[-1 1])
% %title('KK 21'), axis xy square; colorbar
% %subplot(223), imagesc(X(1,ind),Y(ind,1),KK(ind,ind,1,2), 0.1*[-1 1])
% %title('KK 12'), axis xy square; colorbar
% %subplot(224), imagesc(X(1,ind),Y(ind,1),KK(ind,ind,2,2), [0 1])
% %title('KK 22'), axis xy square; colorbar
% 
% %figure(8);
% %subplot(221), imagesc(X(1,ind),Y(ind,1), KK(ind,ind,1,1) - C(ind,ind,1,1))
% %title('C11 - KK 11'), axis xy square; colorbar
% %subplot(222), imagesc(X(1,ind),Y(ind,1), KK(ind,ind,2,1) - C(ind,ind,2,1))
% %title('C21 - KK 21'), axis xy square; colorbar
% %subplot(223), imagesc(X(1,ind),Y(ind,1), KK(ind,ind,1,2) - C(ind,ind,1,2))
% %title('C12 - KK 12'), axis xy square; colorbar
% %subplot(224), imagesc(X(1,ind),Y(ind,1), KK(ind,ind,2,2) - C(ind,ind,2,2))
% %title('C22 - KK 22'), axis xy square; colorbar
% 
% 
% %% Generate random field
% n = 250;
% tic
% F = gen_randomfield_fft(K_fft, n);
% toc
% % 10 min for n = 100, sz = 1024
% 
% F2=(zeros(size(F)));
% for ii=1:n
%     Fy_vec = F(:,:,2,ii);
%     varFy = var(Fy_vec(:));
%     F2(:,:,2,ii)=Fy_vec./sqrt(varFy);
% 
%     Fx_vec = F(:,:,1,ii);
%     varFx = var(Fx_vec(:));
%     F2(:,:,1,ii)=Fx_vec./sqrt(varFx);
% end
% 
% F = (F2*sigma_r);

%figure(9);
%subplot(221), imagesc(F(:,:,1,1), [-4 4]*sigma_r)
%title('Fx'), axis xy square; colorbar
%subplot(222), imagesc(F(:,:,2,1), [-4 4]*sigma_r)
%title('Fy'), axis xy square; colorbar
%subplot(223), imagesc(F(:,:,1,2), [-4 4]*sigma_r)
%title('Fx'), axis xy square; colorbar
%subplot(224), imagesc(F(:,:,2,2), [-4 4]*sigma_r)
%title('Fy'), axis xy square; colorbar
%%
%% Applying the turbulence on an image and henhancing contrast
load F.mat
Fnum = 250; %The field number which we will use from now
str_Img = '/ssd_scratch/cvit/shyam.nandan/AirTurbulence_Dataset_v2/Imagenet/';
str_tImg = '/ssd_scratch/cvit/shyam.nandan/AirTurbulence_Dataset_v2/Imagenet_turb/';
start_path = '/ssd_scratch/cvit/shyam.nandan/ImageNet/train/';
allSubFolders = genpath(start_path);
% % Parse into a cell array.
remain = allSubFolders;
listOfFolderNames = {};
Inc = 0;
while Inc<1000
    [singleSubFolder, remain] = strtok(remain, ':');
    if isempty(singleSubFolder)
        break;
    end
    listOfFolderNames = [listOfFolderNames singleSubFolder];
    Inc = Inc + 1;
end
numberOfFolders = length(listOfFolderNames);
% Process all image files in those folders.
Total_Image = 0;
for k = 1 : numberOfFolders
    clear field_n;
    clear thisFolder;
    clear filePattern
    clear baseFileNames;
    clear numberOfImageFiles;
    
    thisFolder = listOfFolderNames{k};
    filePattern = sprintf('%s/*.JPEG', thisFolder);
    baseFileNames = dir(filePattern);
    numberOfImageFiles = length(baseFileNames);
    k
    if Total_Image > 400000
        break;
    end
    if numberOfImageFiles >= 1
        Inc1 = 0;
        for f = 1 : numberOfImageFiles
            k
            try
                filename = fullfile(thisFolder, baseFileNames(f).name);
                I = imread(filename);
                %Ir = imresize(I,[1024 1024]);
                field_n = randi([1,200],1);
                t_field = F(:,:,:,field_n);
                ISize = 512;
                I = imresize(I, [ISize ISize]);
                CropTurbulentField = t_field(randi(1024-ISize+1)+(0:ISize-1),randi(1024-ISize+1)+(0:ISize-1),:);
                It = turbulence(I,CropTurbulentField);
                It = imresize(It, [224 224]);
                %It = turbulence(Ir, t_field);
                %I = imresize(I,2);
                %It =  imresize(It,[size(I,1) size(I,2)]);
                
                if Inc1 < 6
                    %strcat(str_tImg,'val/',baseFileNames(f).name)
                    imwrite(It,strcat(str_tImg,'val/',baseFileNames(f).name),'jpeg');
                    imwrite(I,strcat(str_Img,'val/',baseFileNames(f).name),'jpeg');
                else
                    %strcat(str_tImg,'train/',baseFileNames(f).name)
                    imwrite(It,strcat(str_tImg,'train/',baseFileNames(f).name),'jpeg');
                    imwrite(I,strcat(str_Img,'train/',baseFileNames(f).name),'jpeg');
                end
                Inc1 = Inc1 + 1;
                Total_Image = Total_Image + 1
                
                if Inc1 > 406
                    break;
                end
            catch
                continue;
            end
        end
    else
        fprintf('Folder %s has no image files in it.\n', thisFolder);
    end
end
% length(srcFiles)
%    filename = strcat('/ssd_scratch/cvit/shyam.nandan/train/n01774750/',srcFiles(i).name);
%    I = imread(filename);
%    I = imresize(I,[1024 1024]); 
%    It = turbulence(I, F(:,:,:,Fnum));
    
%    I = imresize(I,[224 224]); 
%    It =  imresize(It,[224 224]);
%    imwrite(I,strcat(str_Img,srcFiles(i).name),'jpeg');
%    imwrite(It,strcat(str_tImg,srcFiles(i).name),'jpeg');
%end


