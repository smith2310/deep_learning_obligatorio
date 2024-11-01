import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        """
        Inicializa la función de pérdida Dice.

        Args:
            smooth (float): Parámetro para evitar divisiones por cero. Se le suma tanto al numerador
                            como al denominador para estabilizar el cálculo.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Calcula la Dice Loss entre las predicciones y los objetivos.

        Args:
            preds (torch.Tensor): Tensor con las predicciones del modelo. De forma típica en el rango [0, 1] tras aplicar sigmoid.
            targets (torch.Tensor): Tensor con las etiquetas reales (máscara binaria) en el mismo rango y dimensiones.

        Returns:
            torch.Tensor: El valor de la Dice Loss calculada.
        """
        # Aplanar los tensores para calcular el índice de Dice en todo el batch
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Cálculo del índice de Dice
        intersection = (preds * targets).sum()
        dice_coef = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        # Calcula la Dice Loss
        return 1 - dice_coef
