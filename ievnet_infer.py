''' 
This notebook assumes the following package versions:

MONAI version: 0.3.0rc4
Python version: 3.6.10 |Anaconda, Inc.| (default, May  8 2020, 02:54:21)  [GCC 7.3.0]
OS version: Linux (4.15.0-144-generic)
Numpy version: 1.19.1
Pytorch version: 1.7.0a0+8deb4fe
MONAI flags: HAS_EXT = False, USE_COMPILED = False

Optional dependencies:
Pytorch Ignite version: 0.4.2
Nibabel version: 3.1.1
scikit-image version: 0.15.0
Pillow version: 7.2.0
Tensorboard version: 1.15.0+nv
gdown version: 3.12.2
TorchVision version: 0.8.0a0
ITK version: 5.1.1
tqdm version: 4.50.0

For details about installing the optional dependencies, please visit:
    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies

Later MONAI/PyTorch versions are likely to have slight changes in syntax.
'''


import logging
import os
import shutil
import sys
import time
import tempfile
from glob import glob

import argparse

import numpy as np
import torch

import monai
from monai.data import ITKReader
from monai.transforms import (
    Compose, 
    CenterSpatialCrop,
    LoadNifti,
    LoadImage,
    AddChannel,
    AsChannelFirst,
    AsChannelLast,
    SpatialPad,
    ScaleIntensity,
    ToTensor,
)

monai.config.print_config()

class IEVnetInferer():
    '''
        instantiate ievnet=IEVnetInferer()
        call ievnet.predict(filenames_in, filenames_out) for prediction of a list of volumes
        assumes input and writes output volumes of dimensions [200,150,100] at 0.2mm resolution
    '''
    def __init__(self, device='cpu'):
        # using MONAI Unet implementation for cuda optimizations
        # parametrize UNet like VNet (see: https://arxiv.org/abs/1606.04797): 
        #   - 4x downsampling
        #   - 16/32/64/128/256 filter channels, 
        #   - down/up-convolutions with stride 2 instead of max-pooling/up-pooling
        #   - number of residual units in each layer: 2 
        #   - Dice loss
        if device=='cpu':
            map_location=torch.device('cpu')
        elif 'cuda' in device:
            if not torch.cuda.is_available():
                print('Cuda not available, falling back to CPU inference.')
                device='cpu'
                map_location=torch.device('cpu')
            else:
                map_location=torch.device(device)
        
        print('Loading IEVNet model...')
        self.model = monai.networks.nets.UNet(
                        dimensions=3,
                        in_channels=1,
                        out_channels=1,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2),
                        dropout=0.5,
                        num_res_units=2).to(device)
        self.model.load_state_dict(torch.load('best_metric_model.pth', 
                                               map_location=map_location))
        self.model.eval()
        
        self.transforms = Compose([LoadNifti(image_only=True),
                                   AddChannel(),
                                   SpatialPad(spatial_size=[208, 160, 112]),
                                   ScaleIntensity(),
                                   AddChannel(),
                                   ToTensor(),])
        
        # post-processing: center-cropping back to 200,150,100 voxels
        self.cropper = CenterSpatialCrop([200, 150, 100])
        
        print('Done loading model - IEVNet ready.')
    
    def predict(self, filenames_in, filenames_out):
        # create dataset and loader on the fly
        dataset = monai.data.ArrayDataset(filenames_in, img_transform=self.transforms)
        saver = monai.data.NiftiSaver(resample=False)
        
        with torch.no_grad():
            # run seg inference (only one file in dataset)
            for idx, img in enumerate(dataset): 
                print('Infering volume %d of %d (%s)'%(idx+1, len(filenames_in), filenames_in[idx]))
                # predict
                pred_raw = self.model(img)
                print('Inference done.')
                # apply sigmoid
                sigmoid = torch.nn.Sigmoid()
                pred_sigmoid = sigmoid(pred_raw)
                # post-processing of results
                pred = self.cropper(torch.squeeze(pred_sigmoid,0))
                # export
                # get original ITK image header info
                print('Saving.')
                reader = monai.data.ITKReader()
                img = reader.read(filenames_in[idx])
                img_arr, img_hdr = reader.get_data(img)
                meta_data = img_hdr
                meta_data['filename_or_obj'] = filenames_out[idx]
                #saver.save(pred,meta_data)
                monai.data.write_nifti(pred.squeeze().cpu().numpy(), filenames_out[idx], affine=meta_data['affine'], resample=False)
                print('Done.')

def main(filenames_in, filenames_out, device='cpu'):
    ievnet = IEVnetInferer(device=device)
    ievnet.predict(filenames_in, filenames_out)

if __name__ == '__main__':
    # sample inference on two example volumes
    parser = argparse.ArgumentParser(description = 'Program for IEVnet inference.')
    parser.add_argument('--filenames_in', dest='filenames_in', nargs='+', type=str, required=True,
                        help="Filepath(s) to inner ear volumes (crops). Assumes input and writes output volumes of dimensions [200,150,100] at 0.2mm resolution. Several filepaths are separated with a single space.")
    parser.add_argument('--filenames_out', dest='filenames_out', nargs='+', type=str, required=True,
                        help="Output filepath(s) to inner ear segmentations. Will be written as .nii.gz volumes of dimensions [200,150,100] at 0.2mm resolution. Several filepaths are separated with a single space.")
    parser.add_argument('--device', dest='device', choices=['cpu','cuda'], type=str, default='cpu', help='Device for inference. Can be "cpu" or "cuda" (i.e. GPU).')
    args = parser.parse_args()
    main(args.filenames_in, args.filenames_out, args.device)

