from os.path import join
import glob
import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
import tifffile as tiff
from PIL import Image
import numpy as np
import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random
from functools import reduce
from utils.data_utils import imagesc
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce


def get_transforms(opt, additional_targets, need=('train', 'test')):
    transformations = {}
    if opt.rotate:
        rotate_p = 1
    else:
        rotate_p = 0
    if 'train' in need:
        transformations['train'] = A.Compose([
            A.CenterCrop(height=opt.precrop, width=opt.precrop, p=1.),
            A.Resize(opt.resize, opt.resize),
            A.augmentations.geometric.rotate.Rotate(limit=45, p=rotate_p, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(height=opt.cropsize, width=opt.cropsize, p=1.),
            #A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=400),
            #A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    if 'test' in need:
        transformations['test'] = A.Compose([
            A.Resize(opt.resize, opt.resize),
            # may have problem here _----------------------------------
            # A.CenterCrop(height=crop_size, width=crop_size, p=1.),
            A.CenterCrop(height=opt.cropsize, width=opt.cropsize, p=1.), 
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)
    return transformations


class PairedImageDataset(data.Dataset):
    """Dataset for loading and processing paired images from different directories.

    The dataset handles loading, preprocessing, and augmentation of image pairs that
    share filenames across different source directories.
    """

    def __init__(
            self,
            root: str,
            path: str,
            config: dict,
            mode: str,
            labels: Optional[List[int]] = None,
            transforms_fn: Optional[callable] = None,
            return_filenames: bool = False,
            subset_indices: Optional[List[int]] = None
    ):
        super().__init__()
        self.config = config
        self.mode = mode
        self.return_filenames = return_filenames
        self.subset_indices = subset_indices

        # Set up paths
        self.paths = [Path(root) / x for x in path.split('_')]
        self._validate_paths()

        # Find common images across all directories
        self.image_names = self._get_common_images()
        self.subjects = {x: [x] for x in self.image_names}
        self.orders = sorted(self.subjects.keys())

        # Configure image processing parameters
        self.config.resize = self._get_resize_value()
        self.config.crop_size = self.config.cropsize or self.resize
        self.transforms = transforms_fn or self._get_default_transforms()
        self.labels = labels or [0] * len(self.image_names)

    def _validate_paths(self) -> None:
        """Verify all paths exist and are directories."""
        for path in self.paths:
            if not path.is_dir():
                raise ValueError(f"Invalid directory path: {path}")

    def _get_common_images(self) -> List[str]:
        """Find image names that exist in all directories."""
        image_lists = [set(p.glob('*')) for p in self.paths]
        return sorted(list(reduce(set.intersection, [set(item.name for item in images) for images in image_lists])))

    def _get_resize_value(self) -> int:
        """Determine resize value from config or first image."""
        if self.config.resize:
            return self.config.resize
        first_image_path = self.paths[0] / self.image_names[0]
        try:
            return self._load_image(first_image_path).shape[1]
        except Exception as e:
            raise RuntimeError(f"Failed to determine image size: {e}")

    def _get_precrop_value(self) -> int:
        """Determine precrop value from config or first image."""
        if self.config.precrop:
            return self.config.precrop

        first_image_path = self.paths[0] / self.image_names[0]
        try:
            return self._load_image(first_image_path).shape[1]
        except Exception as e:
            raise RuntimeError(f"Failed to determine image size: {e}")

    def _get_default_transforms(self) -> callable:
        """Create default transform pipeline."""
        additional_targets = {str(i).zfill(4): 'image' for i in range(1, len(self.paths))}
        return get_transforms(
            opt=self.config,
            additional_targets=additional_targets
        )[self.mode]

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess single image."""
        try:
            img = tiff.imread(str(path))
        except:
            img = np.array(Image.open(path))

        img = img.astype(np.float32)

        ## Apply threshold
        #if self.config.threshold > 0:
        #    img = np.clip(img, None, self.config.threshold)

        # Normalize
        img = self._normalize_image(img, method=self.config.nm)

        return img

    def _normalize_image(self, img: np.ndarray, method) -> np.ndarray:
        """Normalize image based on configuration."""
        if method == '01':  # 0 to 1
            img = (img - img.min()) / (img.max() - img.min())
        elif method == '11':  # -1 to 1
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 2 - 1
        elif method == '00':
            pass
        return img

    def _prepare_batch(self, filenames: List[str]) -> Dict[str, np.ndarray]:
        """Load and prepare batch of images."""
        batch = {}
        for i, filename in enumerate(filenames):
            key = 'image' if i == 0 else str(i).zfill(4)
            batch[key] = self._load_image(Path(filename))
        return batch

    def __len__(self) -> int:
        return len(self.subset_indices or self.orders)

    def __getitem__(self, idx: int) -> Dict[str, Union[List[torch.Tensor], int, List[str]]]:
        """Get item by index and process image data.
        
        Returns:
            Dict containing:
            - img: List of processed image tensors in (C, H, W, D) format
            - labels: Label for the current index
            - filenames: List of source filenames
        """
        if self.subset_indices is not None:
            idx = self.subset_indices[idx]

        # Get all slices for subject
        subject_slices = sorted(self.subjects[self.orders[idx]])
        filenames = [str(path / slice_name) for path in self.paths for slice_name in subject_slices]

        # Prepare and augment batch
        batch = {k: np.transpose(v, (1, 2, 0)) for k, v in self._prepare_batch(filenames).items()}
        augmented = self.transforms(**batch)
        
        # Process outputs with proper dimensions (C, H, W, D)
        outputs = [augmented.get('image')] + [augmented.get(str(i).zfill(4)) for i in range(1, len(self.paths))]
        outputs = [x.permute(1, 2, 0).unsqueeze(0) for x in outputs]

        return {
            'img': outputs,
            'labels': self.labels[idx],
            'filenames': filenames
        }


if __name__ == '__main__':

    from dotenv import load_dotenv
    import argparse
    import matplotlib.pyplot as plt
    load_dotenv('env/.t09')

    parser = argparse.ArgumentParser(description='Visualize paired image dataset')
    parser.add_argument('--dataset', type=str, default='womac4', help='Dataset to use (womac4, dess0, brain)')
    parser.add_argument('--direction', type=str, default=None, help='Override dataset direction')
    parser.add_argument('--rotate', action='store_true', help='Enable rotation augmentation')
    parser.add_argument('--idx', type=int, default=2, help='Index to visualize')
    parser.add_argument('--slice', type=int, default=None, help='Slice to visualize for 3D data')
    parser.add_argument('--port', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    parser.add_argument('--mode', type=str, default='dummy')
    opt = parser.parse_args()

    # Predefined dataset configurations
    dataset_configs = {
        'womac4': {
            'root': '/media/ExtHDD01/Dataset/paired_images/womac4/train/',
            'direction': 'a3d',
            'precrop': 380,
            'cropsize': 384,
            'resize': 384,
            'nm': '11',
            'rotate': True,
            'rgb': False,
            'threshold': 0
        },
        'dess0': {
            'root': '/media/ExtHDD01/Dataset/paired_images/dess0/',
            'direction': 'ori_ori',
            'precrop': 384,
            'cropsize': 256,
            'resize': 0,
            'nm': '00',
            'rotate': True,
            'rgb': False,
            'threshold': 0
        },
        'brain': {
            'root': '/media/ghc/Ghc_data3/Chu_full_brain/enhance_test/ROI/train/',
            'direction': 'dpm3D_dpmm3D',
            'precrop': 256,
            'cropsize': 256,
            'resize': 0,
            'nm': '00',
            'rotate': True,
            'rgb': False,
            'threshold': 0
        },
    }
    
    # Load the selected dataset configuration
    if opt.dataset in dataset_configs:
        config = dataset_configs[opt.dataset]
        for key, value in config.items():
            setattr(opt, key, value)
    else:
        print(f"Dataset {opt.dataset} not found in configurations")
        exit(1)
    
    # Override direction if specified
    if opt.direction is not None:
        opt.direction = opt.direction
    
    print(f"Testing dataset: {opt.dataset}")
    print(f"Root: {opt.root}")
    print(f"Direction: {opt.direction}")
    
    # Create the dataset
    d = PairedImageDataset(root=opt.root, path=opt.direction, config=opt, mode='test')
    print(f"Dataset size: {len(d)}")
    
    # Get a sample and visualize
    x = d.__getitem__(opt.idx)
    print(f"Image shape: {x['img'][0].shape}")
    print('min:', x['img'][0].min(), 'max:', x['img'][0].max())
    
    # For 3D data, visualize a specific slice or middle slice
    if len(x['img'][0].shape) == 4:  # 3D data (C, H, W, D)
        slice_idx = opt.slice if opt.slice is not None else x['img'][0].shape[3] // 2
        for i, img in enumerate(x['img']):
            imagesc(img[0, :, :, slice_idx])
    else:  # 2D data
        for i, img in enumerate(x['img']):
            imagesc(img[0])
    
    print(f"Filenames: {x['filenames']}")

