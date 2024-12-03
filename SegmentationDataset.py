import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T 
from typing import Callable, Optional

class SegmentacionDataset(Dataset):
    """
    Dataset que se encarga de cargar los datos y resultados para el dataset provisto en Kaggle
    """

    def __init__(
            self,
            root_dir: Path,
            load_mask: bool,
            image_format: str = 'RGB',
            mask_format: str = 'L',
            x_transform: Optional[Callable] = None,
            y_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.load_mask = load_mask
        self.image_format = image_format
        self.mask_format = mask_format
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.image_names = os.listdir(root_dir/'images')

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.get_name(index)
        image_path = self.root_dir / 'images' / image_name
        image = Image.open(image_path).convert(self.image_format)

        if self.x_transform:
            image = self.x_transform(image)
            
        if self.load_mask:
            mask_path = self.root_dir / 'masks' / image_name
            mask = Image.open(mask_path).convert(self.mask_format)
            if self.y_transform:
                mask = self.y_transform(mask)
            return image, mask
        else:
            return image
        
    def get_name(self, index):
        return self.image_names[index]
