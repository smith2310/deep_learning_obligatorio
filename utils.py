import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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

    def __call__(self, val_loss):
        if val_loss >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
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
    do_early_stopping=True,
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
        Tuple[List[float], List[float]]: Una tupla con dos listas, la primera con el error de entrenamiento de cada época y la segunda con el error de validación de cada época.

    """
    epoch_train_errors = []
    epoch_val_errors = []
    best_val_loss = float('inf')
    best_model_weights = None

    if do_early_stopping:
        early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
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

        if do_early_stopping:
            early_stopping(val_loss)

        if log_fn is not None:
            if (epoch + 1) % log_every == 0:
                log_fn(epoch, train_loss, val_loss)

        if do_early_stopping and early_stopping.early_stop:
            print(
                f"Detener entrenamiento en la época {epoch}, la mejor pérdida fue {early_stopping.best_score:.5f}"
            )
            break

    # Cargar los mejores pesos al modelo al final del entrenamiento
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return epoch_train_errors, epoch_val_errors



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


def model_classification_report(model, dataloader, device, nclasses):
    """
    Returns:
        (accuracy, f1_score, precision, recall) promedio
    """
    # Evaluación del modelo
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Reporte de clasificación
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[str(i) for i in range(nclasses)],
        output_dict=True
    )
    averages = report['weighted avg']
    return (report['accuracy'], averages['f1-score'], averages['precision'], averages['recall'])


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

def calculate_mean_and_std(dataset, cache_file_name = 'images_data_estimation.txt'):
    """
    Calcula la media y la desviación estándar de un conjunto de datos.

    Args:
        dataset (torch.utils.data.Dataset): Conjunto de datos.
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

    # Calcular la media y la desviación estándar
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for image, _ in dataset:
        mean += torch.mean(image, dim=[1, 2])
        std += torch.std(image, dim=[1, 2])

    mean /= len(dataset)
    std /= len(dataset)

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