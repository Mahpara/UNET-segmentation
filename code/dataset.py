import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torch

NUM_CLASSES = 13

class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.images=os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_raw=np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask_raw)
            image = augmentations["image"]
            mask_raw = augmentations["mask"]
        
        one_hot_matrix = np.eye(NUM_CLASSES)
        output = np.empty(mask_raw.shape + (NUM_CLASSES,))
        for i in range(NUM_CLASSES):
            output[mask_raw == i] = one_hot_matrix[i]
        mask = output.transpose((2,0,1))
            
        return image, mask