import random
import numpy as np
from torchvision.transforms import functional as T
from PIL import Image
import cv2

import numpy as np
from PIL import Image

class ElasticTransform:
    def __init__(self, alpha=1, sigma=10, grid_size=(3, 3), seed=None):
        """
        Parámetros:
        - alpha: magnitud de las deformaciones (cuánto varían los desplazamientos).
        - sigma: desviación estándar de la distribución normal que generará las deformaciones.
        - grid_size: tamaño de la cuadrícula de desplazamientos (por ejemplo, 3x3).
        - seed: semilla para la generación de números aleatorios.
        """
        self.alpha = alpha
        self.sigma = sigma
        self.grid_size = grid_size
        self.seed = seed

    def __call__(self, image: Image.Image):
        """
        Aplica la transformación elástica a una imagen PIL.
        """

        if self.seed is not None:
            current_seed = np.random.get_state()
            np.random.seed(self.seed)
            
        # Convertir la imagen a un arreglo numpy
        image_array = np.array(image)  # Convertir la imagen PIL a numpy array
        
        # Verificar el modo de la imagen
        is_rgb = len(image_array.shape) == 3  # Si tiene tres canales, es RGB, si no es L (escala de grises)
        
        if not is_rgb:
            h, w = image_array.shape  # En modo 'L', solo hay una dimensión de color
        else:
            h, w, c = image_array.shape  # En modo 'RGB', hay tres dimensiones: altura, ancho y canales

        # Generar desplazamientos aleatorios en una cuadrícula
        grid_h, grid_w = self.grid_size
        displacement = np.random.normal(loc=0, scale=self.sigma, size=(grid_h + 1, grid_w + 1, 2))
        displacement *= self.alpha

        # Crear una malla de coordenadas
        y, x = np.meshgrid(np.arange(h), np.arange(w))
        coords = np.stack((x, y), axis=-1)

        # Suavizar los desplazamientos en la malla utilizando un filtro Gaussiano
        displacement_map = self._smooth_displacement(displacement, (h, w))

        # Generar coordenadas deformadas
        displaced_coords = coords + displacement_map

        # Asegurarse de que las coordenadas no se salgan de los límites
        displaced_coords[..., 0] = np.clip(displaced_coords[..., 0], 0, h - 1)
        displaced_coords[..., 1] = np.clip(displaced_coords[..., 1], 0, w - 1)

        # Realizar la interpolación para obtener la imagen deformada
        deformed_image = self._interpolate_image(image_array, displaced_coords, is_rgb)

        # Convertir el resultado de nuevo a PIL (Imagen de vuelta a formato PIL)
        deformed_image = Image.fromarray(deformed_image.astype(np.uint8))

        if self.seed is not None:
            np.random.set_state(current_seed)

        return deformed_image

    def _smooth_displacement(self, displacement, size):
        """
        Suaviza los desplazamientos utilizando un filtro Gaussiano.
        """
        displacement_map = np.zeros((size[0], size[1], 2), dtype=np.float32)

        # Calcular los pasos de la cuadrícula
        grid_h, grid_w = self.grid_size
        step_h = size[0] // (grid_h + 1)
        step_w = size[1] // (grid_w + 1)

        # Asegurarse de que los pasos no sean cero
        step_h = max(step_h, 1)
        step_w = max(step_w, 1)

        # Interpolación de los desplazamientos en la cuadrícula
        for i in range(grid_h + 1):
            for j in range(grid_w + 1):
                y_start = i * step_h
                y_end = (i + 1) * step_h
                x_start = j * step_w
                x_end = (j + 1) * step_w

                displacement_map[y_start:y_end, x_start:x_end, 0] = displacement[i, j, 0]
                displacement_map[y_start:y_end, x_start:x_end, 1] = displacement[i, j, 1]

        return displacement_map

    def _interpolate_image(self, image_array, displaced_coords, is_rgb):
        """
        Interpola la imagen original utilizando las coordenadas deformadas.
        """
        h, w = image_array.shape[:2]  # Obtener las dimensiones de altura y ancho
        if is_rgb:
            c = image_array.shape[2]  # Si es RGB, hay tres canales
            deformed_image = np.zeros((h, w, c), dtype=np.float32)
        else:
            deformed_image = np.zeros((h, w), dtype=np.float32)  # Si es L, solo un canal

        # Realizar la interpolación para cada píxel
        for i in range(h):
            for j in range(w):
                x, y = displaced_coords[i, j]
                x, y = int(x), int(y)

                if 0 <= x < h and 0 <= y < w:
                    if is_rgb:
                        deformed_image[i, j, :] = image_array[x, y, :]
                    else:
                        deformed_image[i, j] = image_array[x, y]

        return deformed_image




    
class RandomBrightnessContrast:
    """Ajustar brillo y contraste aleatoriamente."""
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        if random.random() < 0.5:
            img = T.adjust_brightness(img, 1 + random.uniform(-self.brightness, self.brightness))
        if random.random() < 0.5:
            img = T.adjust_contrast(img, 1 + random.uniform(-self.contrast, self.contrast))
        return img
