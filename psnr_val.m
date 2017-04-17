clear

scale = 2;
sr_path = '/home/tang/NTIRE/DIV2K_valid_LR_bicubic/SRX2';
gt_path = '/home/tang/NTIRE/DIV2K_valid_HR';

sr_filepaths = dir(fullfile(sr_path,'*.png'));
gt_filepaths = dir(fullfile(gt_path,'*.png'));

for i = 1 : 100
    G = fullfile(sr_path, sr_filepaths(i).name);
    F = fullfile(gt_path, gt_filepaths(i).name);
    psnr(i) = NTIRE_PeakSNR_imgs(F, G, scale);
end

res = sum(psnr)/length(psnr)