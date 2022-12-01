path = '/home/shyam.nandan/data/train/o/';
srcFiles = dir(strcat(path,'*.jpg'));  % the folder in which ur images exists
for i = 1 : length(srcFiles)
        
    filename = strcat(path,srcFiles(i).name);
    I = imread(filename);
    %if size(I,3)==1
    i
    %   I = cat(3, I, I, I);
    %   imwrite(I,strcat(path,srcFiles(i).name),'jpeg');
    %end
    I = imresize(I,[256,256]);
    imwrite(I,strcat(path,srcFiles(i).name),'jpeg');
end
