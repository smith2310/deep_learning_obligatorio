import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetNode(nn.Module):
    """
    Una instancia de UNetNode representa de forma genérica cualquier nodo de la arquitectura U-Net,
    ya sea el lado izquierdo, el derecho o la base.
    """
    def __init__(self, in_channels, out_channels, last_operation):
        super(UNetNode, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.last_operation = last_operation

    def forward(self, x):
        """
        Calcula el forward del nodo
        Returns:
            Tuple: (previous_output, output), donde {previous_output} es la salida de las 2 primeras
            convoluciones antes de aplicar {last_operation} y output es el resultado de aplicar {last_operation}
            a {previous_last_output}
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        layer_output = x
        vertical_output = self.last_operation(x)
        return (layer_output, vertical_output)

class UNetLayer(nn.Module):
    """
    Una instancia de UNetLayer representa una layer de la arquitectura U-Net, desde la que se encuentra como
    entrada más arriba hasta la que se tiene en la base.
    """
    def __init__(self, left_node, right_node, next_layer):
        super(UNetLayer, self).__init__()
        self.left_node = left_node
        self.right_node = right_node
        self.next_layer = next_layer #Representa la layer de más abajo, en caso de ser la base será None

    def forward(self, x):
        """
        Calcula el forward para el layer, llamando al layer siguiente y eventualmente
        concatenando el resultado del {left_node} con el resultado del {next_layer}
        """
        previous_output, output = self.left_node(x)
        if self.next_layer:
            #Asumimos que los tamaños de los tensores van a ser los mismos del left y next_layer
            next_output = self.next_layer(output)
            concatenated_output = torch.cat((previous_output, next_output), dim=1)
            return self.right_node(concatenated_output)
        else:
            return output

class UNet(nn.Module):
    """
    Una instancia de UNet representa todo el modelo con las diferentes layers, donde la primera y más "de arriba" es la entrada
    y la salida. El constructor recibe como parametro una lista de enteros con los in_channels de todas las layers,
    donde el primero representa el in_channels de la entrada al modelo, el segundo es el in_channels del top_layer, luego el 
    in_channels, de la siguiente layer, hasta que el último representa la cantidad de in_channels de la base de la U-Net.
    """
    def __init__(self, in_channels_list):
        super(UNet, self).__init__()
        layers = []
        for i in range(len(in_channels_list) - 1):
            left_node = UNetNode(in_channels_list[i], in_channels_list[i + 1], nn.MaxPool2d(kernel_size=2, stride=2))
            right_node = UNetNode(in_channels_list[i + 1] * 2, in_channels_list[i], nn.ConvTranspose2d(in_channels_list[i + 1], in_channels_list[i], kernel_size=2, stride=2))
            next_layer = layers[-1] if layers else None
            layers.append(UNetLayer(left_node, right_node, next_layer))
        self.top_layer = layers[0]

    def forward(self, x):
        return self.top_layer(x)