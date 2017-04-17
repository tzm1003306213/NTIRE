function im2h5( image_paths, input_size, stride, scale, created_flag, savepath )

label_size = input_size * scale;
chunksz = 5;
count = 0;

for i = 1 : length(image_paths)
    
    info = imfinfo(image_paths{i});
    
    if strcmp(info.ColorType, 'truecolor')
        
        image = imread(image_paths{i});
        image = image(:,:,[3 2 1]);

        im_label = modcrop(image, scale);
        im_input = imresize(im_label, 1/scale, 'bicubic');
        [hei,wid,~] = size(im_input);
        
        if hei>=input_size && wid>=input_size
            
            for x = 1 : stride : hei-input_size+1
                for y = 1 :stride : wid-input_size+1
                    subim_input = im_input(x : x+input_size-1, y : y+input_size-1, :);
                    in_x = ((x-1)*scale)+1;
                    in_y = ((y-1)*scale)+1;
                    subim_label = im_label(in_x : in_x+label_size-1, in_y : in_y+label_size-1, :);

                    count=count+1;
                    subim_input = single(subim_input);
                    subim_label = single(subim_label);            
                    data(:, :, :, count) = subim_input;
                    label(:, :, :, count) = subim_label;
                    
                end
            end
            
        end
    end
end

data = permute(data,[3,2,1,4]);
label = permute(label,[3,2,1,4]);

%% writing to HDF5
%
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read = (batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);


end

