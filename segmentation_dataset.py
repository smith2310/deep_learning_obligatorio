import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from custom_transformations import ElasticTransform, RandomBrightnessContrast
from utils import calculate_mean_and_std

class SegmentacionDataset(Dataset):
    """
    Dataset que se encarga de cargar los datos y resultados para el dataset provisto en Kaggle
    Args:
        root_dir (Path): Directorio raiz donde se encuentran las imagenes y mascaras
        load_mask (bool): Si se desea cargar las mascaras
        x_transform (Optional[Callable]): Transformaciones base para las imagenes
        y_transform (Optional[Callable]): Transformaciones base para las mascaras
    """ 
    def __init__(
            self,
            root_dir: Path,
            image_size: tuple,
            load_mask: bool
    ):
        super().__init__()
        self.root_dir = root_dir
        self.load_mask = load_mask
        images_path = root_dir / 'images'
        self.image_names = os.listdir(images_path)
        mean_std_cache = "mean_std_cache.txt"
        mean, std = calculate_mean_and_std(images_path, mean_std_cache)
        self.mean = mean
        self.std = std
        # Tenemos que asegurarnos que aplicamos las mismas transformaciones a la imagen y a la máscara
        self.elastic_transform = ElasticTransform()
        self.image_transforms = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip(p=0.5),
            # T.RandomRotation(degrees=45),
            self.elastic_transform,
            RandomBrightnessContrast(brightness=0.2, contrast=0.2),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean, std)
        ])
        self.mask_transforms = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip(p=0.5),
            # T.RandomRotation(degrees=45),
            self.elastic_transform,
            T.ToImage(),
            T.ToDtype(torch.float32)
        ])
        

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.get_name(index)
        image_path = self.root_dir / 'images' / image_name
        image = Image.open(image_path).convert('RGB')

        #Nos aseguramos que aplicamos las mismas transformaciones a la imagen y a la máscara de forma consistente 
        #usando el mismo SEED, para esto generamos un SEED aleatorio y lo fijamos y luego restablecemos el SEED actual
        current_seed = torch.initial_seed()
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        self.elastic_transform.seed = seed #Fijamos el SEED en la transformación
        
        #Aplicamos transformaciones
        image = self.image_transforms(image)
        if self.load_mask:
            mask_path = self.root_dir / 'masks' / image_name
            mask = Image.open(mask_path).convert('L')
            mask = self.mask_transforms(mask)
            torch.manual_seed(current_seed) #restauramos el SEED original
            return image, mask
        else:
            torch.manual_seed(current_seed) #restauramos el SEED original
            return image
        
    def get_name(self, index):
        return self.image_names[index]
