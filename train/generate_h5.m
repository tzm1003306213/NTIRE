clear

folder = '/media/tang/RAID0/NTIRE/DIV2K_train_AUG/';

savepath = '/media/tang/RAID0/NTIRE/X2train.h5';
scale = 2;
input_size = 160;
stride = 80;
created_flag = false;

paths = dir(fullfile(folder));
im_paths = paths(3:end);

for i = 1 : length(im_paths)
    image_paths{i} = fullfile(folder, im_paths(i).name);
end

mod_num = 50;
for i = 1 : mod_num: length(im_paths)-rem(length(im_paths),mod_num)-mod_num+1
    
    im2h5( image_paths(i:i+mod_num-1), input_size, stride, scale, created_flag, savepath );
    created_flag = true;
    
end
