# NTIRE
NTIRE 2017 Super-Resolution Challenge - Track 1: Bicubic downscaling - x2 - Track 1: Bicubic downscaling - x3 - Track 1: Bicubic downscaling - x4

## Overview
The test image is augmented using... , and the results are average fusioned. Only single model is tested.

## Requirments
tensorflow 1.0.1
tensorlayer 1.3.11

## Files
- test.py : test file.
- Models' directory: Models/

## How To Use

### Testing
```shell
# --LR_path, path to the low-resolution images; --save_path, path to save the super-resolved images; --model_path, path to model; --aug, low-resolution image augmentation.
# If low-resolution image augmentation is not used, just ignore the "--aug".
# example:
python test.py --LR_path DIV2K_test_LR_bicubic/X2/ --save_path DIV2K_test_SR/SRX2/ --model_path Models/X2 --aug true
python test.py --LR_path DIV2K_test_LR_bicubic/X3/ --save_path DIV2K_test_SR/SRX3/ --model_path Models/X3 --aug true
python test.py --LR_path DIV2K_test_LR_bicubic/X4/ --save_path DIV2K_test_SR/SRX4/ --model_path Models/X4 --aug true
```

