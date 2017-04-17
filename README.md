# NTIRE
NTIRE 2017 Super-Resolution Challenge - Track 1: Bicubic downscaling - x2 - Track 1: Bicubic downscaling - x3 - Track 1: Bicubic downscaling - x4

## Files
- test.py : test file.
- model files: Models/

## How To Use
### Training
```shell
# if start from scratch
python VDSR.py
# if start with a checkpoint
python VDSR.py --model_path ./checkpoints/CHECKPOINT_NAME.ckpt
```
### Testing
```shell
# python test.py --LR_path DIV2K_test_LR_bicubic/X2/ --save_path DIV2K_test_LR_bicubic/SRX2_final/ --model_path tmp/variables-21000 --aug true
python test.py
```
### Plot Result
```shell
# plot the psnr result stored in ./psnr directory
python PLOT.py
```
