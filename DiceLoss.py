import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensores de predicciones (probabilidades, rango [0, 1]) de forma (batch_size, ..., H, W).
            targets: Tensores de etiquetas reales (binarios, valores 0 o 1) de forma (batch_size, ..., H, W).

        Returns:
            Dice Loss.
        """
        # Aplanar tensores a una sola dimensión
        preds = preds.contiguous().view(preds.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        
        intersection = (preds * targets).sum(dim=1)
        preds_sum = preds.sum(dim=1)
        targets_sum = targets.sum(dim=1)

        # Calcular Dice Coefficient
        dice_coeff = (2.0 * intersection + self.epsilon) / (preds_sum + targets_sum + self.epsilon)
        dice_loss = 1 - dice_coeff.mean()

        return dice_loss
