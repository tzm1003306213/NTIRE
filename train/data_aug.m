clear; 
%% To do data augmentation
folder = '/home/tang/NTIRE/DIV2K_train_HR';
savepath = '/media/tang/RAID0/NTIRE/DIV2K_train_AUG/';

filepaths = dir(fullfile(folder,'*.png'));
     
for i = 1 : length(filepaths)
    filename = filepaths(i).name;
    [add, im_name, type] = fileparts(filepaths(i).name);
    image = imread(fullfile(folder, filename));
    
    for flip = 0: 1 : 1
        
        if flip==1
            im_flip = fliplr(image); 
        else
            im_flip = image;
        end
        
        imwrite(im_flip, [savepath im_name, '_flip' num2str(flip) '.bmp']);
        
    end
end
