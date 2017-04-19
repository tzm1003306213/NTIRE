# NTIRE
NTIRE 2017 Super-Resolution Challenge - Track 1: Bicubic downscaling - x2   
NTIRE 2017 Super-Resolution Challenge - Track 1: Bicubic downscaling - x3   
NTIRE 2017 Super-Resolution Challenge - Track 1: Bicubic downscaling - x4   

## Overview
The test image is augmented using left-right flip, up-down flip and left-right and up-down flip, and the results are average fused. 
Due to the limitation of time and resources, only single model is tested. The complete training code will be released later. Details of models will be described in detail in our factsheet file. 

## Requirements
- tensorflow 1.0.1
- tensorlayer 1.3.11
- opencv 2.4
- h5py 2.6
- CUDA 8.0 (optional)
- cudnn 5.1 (optional)

## Files
- test.py : test file
- ./Models : Models' directory
- train_X2.py : training file
- ./train : training data generating files
- ./DIV2K_test_LR_bicubic : Low-resolution images

## How To Use

### Testing
```shell
# Arguments:
# --LR_path, path to the low-resolution images; --save_path, path to save the super-resolved images;   
# --model_path, path to model; --aug, low-resolution image augmentation.
# If low-resolution image augmentation is not used, just ignore the "--aug".
# 
# Example X2:
python test.py --LR_path ./DIV2K_test_LR_bicubic/X2/ --save_path ./DIV2K_test_SR/SRX2/ --model_path ./Models/X2 --aug true
# Example X3:
python test.py --LR_path ./DIV2K_test_LR_bicubic/X3/ --save_path ./DIV2K_test_SR/SRX3/ --model_path ./Models/X3 --aug true
# Example X4:
python test.py --LR_path ./DIV2K_test_LR_bicubic/X4/ --save_path ./DIV2K_test_SR/SRX4/ --model_path ./Models/X4 --aug true
```
# 

