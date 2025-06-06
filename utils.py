import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    classification_report
)


def evaluate(model, criterion, data_loader, device):
    """
    Evalúa el modelo en los datos proporcionados y calcula la pérdida promedio.

    Args:
        model (torch.nn.Module): El modelo que se va a evaluar.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        data_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de evaluación.

    Returns:
        float: La pérdida promedio en el conjunto de datos de evaluación.

    """
    model.eval()  # ponemos el modelo en modo de evaluacion
    total_loss = 0  # acumulador de la perdida
    with torch.no_grad():  # deshabilitamos el calculo de gradientes
        for x, y in data_loader:  # iteramos sobre el dataloader
            x = x.to(device)  # movemos los datos al dispositivo
            y = y.to(device)  # movemos los datos al dispositivo
            output = model(x)  # forward pass
            total_loss += criterion(output, y).item()  # acumulamos la perdida
    return total_loss / len(data_loader)  # retornamos la perdida promedio


class EarlyStopping:
    def __init__(self, patience=5):
        """
        Args:
            patience (int): Cuántas épocas esperar después de la última mejora.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")
        self.val_loss_min = float("inf")
        self.early_stop = False
        self.best_epoch = 0
        self.epoch_counter = 0

    def __call__(self, val_loss):
        self.epoch_counter += 1
        if val_loss >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_epoch = self.epoch_counter
            self.best_score = val_loss
            self.counter = 0


def print_log(epoch, train_loss, val_loss):
    print(
        f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
    )

import copy

def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    patience=5,
    epochs=10,
    log_fn=print_log,
    log_every=1,
):
    """
    Entrena el modelo utilizando el optimizador y la función de pérdida proporcionados.

    Args:
        model (torch.nn.Module): El modelo que se va a entrenar.
        optimizer (torch.optim.Optimizer): El optimizador que se utilizará para actualizar los pesos del modelo.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        train_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de entrenamiento.
        val_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de validación.
        device (str): El dispositivo donde se ejecutará el entrenamiento.
        patience (int): Número de épocas a esperar después de la última mejora en val_loss antes de detener el entrenamiento (default: 5).
        epochs (int): Número de épocas de entrenamiento (default: 10).
        log_fn (function): Función que se llamará después de cada log_every épocas con los argumentos (epoch, train_loss, val_loss) (default: None).
        log_every (int): Número de épocas entre cada llamada a log_fn (default: 1).

    Returns:
        Tuple[List[float], List[float], Int]: Una tupla con dos listas y un numero, la primera lista con el error de entrenamiento de cada época 
        y la segunda lista con el error de validación de cada época y el número de épocas que se ejecutaron.

    """
    epoch_train_errors = []
    epoch_val_errors = []
    best_val_loss = float('inf')
    best_model_weights = None

    early_stopping = EarlyStopping(patience=patience)

    epochs_counter = 0

    for epoch in range(epochs):
        epochs_counter += 1
        model.to(device)
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            batch_loss = criterion(output, y)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        train_loss /= len(train_loader)
        epoch_train_errors.append(train_loss)
        val_loss = evaluate(model, criterion, val_loader, device)
        epoch_val_errors.append(val_loss)

        # Guardar los pesos del mejor modelo basado en la pérdida de validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        early_stopping(val_loss)
            

        if log_fn is not None:
            if (epoch + 1) % log_every == 0:
                log_fn(epoch, train_loss, val_loss)

        if early_stopping.early_stop:
            print(
                f"Detener entrenamiento en la época {epoch}, la mejor pérdida fue {early_stopping.best_score:.5f} en la época {early_stopping.best_epoch}"
            )
            break

    # Cargar los mejores pesos al modelo al final del entrenamiento
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return epoch_train_errors, epoch_val_errors, early_stopping.best_epoch


def final_train(
    model,
    optimizer,
    criterion,
    train_loader,
    device,
    epochs,
    log_fn=None,  # función de logging opcional
    log_every=1,    
):
    """
    Entrena el modelo utilizando el optimizador y la función de pérdida proporcionados.

    Args:
        model (torch.nn.Module): El modelo que se va a entrenar.
        optimizer (torch.optim.Optimizer): El optimizador que se utilizará para actualizar los pesos del modelo.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        train_loader (torch.utils.data.DataLoader): DataLoader que proporciona TODOS los datos de entrenamiento
        device (str): El dispositivo donde se ejecutará el entrenamiento.
        epochs (int): Número de épocas de entrenamiento en total a correr.
        log_fn (function): Función que se llamará después de cada log_every épocas con los argumentos (epoch, train_loss) (default: None).
        log_every (int): Número de épocas entre cada llamada a log_fn (default: 1).
    """
    model.train()  # ponemos el modelo en modo de entrenamiento
    for epoch in range(epochs):
        model.to(device)
        train_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            batch_loss = criterion(output, y)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
        train_loss /= len(train_loader)
        if log_fn and epoch % log_every == 0:
            log_fn(epoch, train_loss)

        if epoch % log_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")



def plot_taining(train_errors, val_errors):
    # Graficar los errores
    plt.figure(figsize=(10, 5))  # Define el tamaño de la figura
    plt.plot(train_errors, label="Train Loss")  # Grafica la pérdida de entrenamiento
    plt.plot(val_errors, label="Validation Loss")  # Grafica la pérdida de validación
    plt.title("Training and Validation Loss")  # Título del gráfico
    plt.xlabel("Epochs")  # Etiqueta del eje X
    plt.ylabel("Loss")  # Etiqueta del eje Y
    plt.legend()  # Añade una leyenda
    plt.grid(True)  # Añade una cuadrícula para facilitar la visualización
    plt.show()  # Muestra el gráfico


def segmentation_classification_report(model, dataloader, device):
    """
    Genera un reporte de clasificación para tareas de segmentación.
    
    Args:
        model: Modelo UNet.
        dataloader: DataLoader con datos de prueba.
        device: Dispositivo (CPU/GPU).

    Returns:
        Diccionario con métricas promedio (accuracy, f1-score, precision, recall).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # Asegúrate de que las etiquetas también estén en el dispositivo
            outputs = model(inputs)
            
            # Binarizar las salidas con un umbral de 0.5
            preds = (outputs > 0.5).int()
            
            # Aplanar las predicciones y etiquetas para usar sklearn
            all_preds.extend(preds.cpu().numpy().ravel())
            all_labels.extend(labels.cpu().numpy().ravel())

    # Convertir a arrays de numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Reporte de clasificación
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["Clase 0", "Clase 1"],
        output_dict=True
    )
    
    # Extraer métricas promedio
    averages = report['weighted avg']
    return report['accuracy'], averages['f1-score'], averages['precision'], averages['recall']


def show_tensor_image(tensor, title=None, vmin=None, vmax=None):
    """
    Muestra una imagen representada como un tensor.

    Args:
        tensor (torch.Tensor): Tensor que representa la imagen. Size puede ser (C, H, W).
        title (str, optional): Título de la imagen. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    # Check if the tensor is a grayscale image
    if tensor.shape[0] == 1:
        plt.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
    else:  # Assume RGB
        plt.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_tensor_images(tensors, titles=None, figsize=(15, 5), vmin=None, vmax=None):
    """
    Muestra una lista de imágenes representadas como tensores.

    Args:
        tensors (list): Lista de tensores que representan las imágenes. El tamaño de cada tensor puede ser (C, H, W).
        titles (list, optional): Lista de títulos para las imágenes. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    num_images = len(tensors)
    _, axs = plt.subplots(1, num_images, figsize=figsize)
    for i, tensor in enumerate(tensors):
        ax = axs[i]
        # Check if the tensor is a grayscale image
        if tensor.shape[0] == 1:
            ax.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
        else:  # Assume RGB
            ax.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
        if titles and titles[i]:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.show()

def calculate_mean_and_std(folder_path, cache_file_name = 'images_data_estimation.txt'):
    """
    Calcula la media y la desviación estándar de un conjunto de datos.

    Args:
        folder_path (String): Ruta de la carpeta que contiene las imágenes.
        cache_file_name (str): Nombre del archivo donde se guardarán los valores de media y desviación estándar,
        y si el archivo existe en lugar de calcular los valores se leerán del archivo.

    Returns:
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]: Media y desviación estándar de los datos.
    """
    # Leer el archivo si ya se ha calculado previamente
    try:
        with open(cache_file_name, 'r') as file:
            line = file.readline()
            mean = tuple(map(float, line.split(',')))
            line = file.readline()
            std = tuple(map(float, line.split(',')))
            print('Valor de la media y desviación estándar leídos del archivo')
            return mean, std
    except FileNotFoundError:
        mean = None
        std = None

    if not os.path.isdir(folder_path):
        raise ValueError(f"La ruta especificada '{folder_path}' no es una carpeta válida.")

    # Acumuladores para la suma y suma de cuadrados
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_pixels = 0

    # Iterar sobre todas las imágenes en la carpeta
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path).convert('RGB') as img:
            image_tensor = torch.tensor(np.array(img) / 255.0).permute(2, 0, 1) # Normalizar los valores de píxeles

        # Sumar las medias por canal
        mean += torch.mean(image_tensor, dim=[1, 2])
        # Sumar las desviaciones estándar por canal
        std += torch.std(image_tensor, dim=[1, 2])
        # Actualizar el número de píxeles
        num_pixels += image_tensor.shape[1] * image_tensor.shape[2]

    # Promediar las medias y desviaciones estándar acumuladas
    mean /= len(os.listdir(folder_path))
    std /= len(os.listdir(folder_path))

    mean = tuple(mean.tolist())
    std = tuple(std.tolist())

    # Guardar los valores en el archivo
    try:
        with open(cache_file_name, 'w') as file:
            file.write(",".join(map(str, mean)))
            file.write("\n")
            file.write(",".join(map(str, std)))
    except FileNotFoundError:
        pass

    return mean, std

def rle_encode(mask):
    pixels = np.array(mask).flatten(order='F')  # Aplanar la máscara en orden Fortran
    pixels = np.concatenate([[0], pixels, [0]])  # Añadir ceros al principio y final
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Encontrar transiciones
    runs[1::2] = runs[1::2] - runs[::2]  # Calcular longitudes
    return ' '.join(str(x) for x in runs)