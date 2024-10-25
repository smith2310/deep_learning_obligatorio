import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetNode(nn.Module):
    """
    Una instancia de UNetNode representa de forma genérica cualquier nodo de la arquitectura U-Net,
    ya sea el lado izquierdo, el derecho o la base.
    """
    def __init__(self, in_channels, out_channels, last_operation, top_layer=False):
        super(UNetNode, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.last_operation = last_operation
        self.top_layer = top_layer

    def forward(self, x):
        """
        Calcula el forward del nodo
        Returns:
            Tuple: (previous_output, output), donde {previous_output} es la salida de las 2 primeras
            convoluciones antes de aplicar {last_operation} y output es el resultado de aplicar {last_operation}
            a {previous_last_output}
        """
        # print(f'UNetNode 1: {x.shape=}')
        x = F.relu(self.conv1(x))
        # print(f'UNetNode 2: {x.shape=}')
        x = F.relu(self.conv2(x))
        horizontal_output = x
        # print(f'UNetNode 3: {layer_output.shape=}')
        vertical_output = self.last_operation(x)
        if self.top_layer:
            vertical_output = torch.sigmoid(vertical_output)
        # print(f'UNetNode 4: {vertical_output.shape=}')
        return (horizontal_output, vertical_output)

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
        # print(f'UNetLayer 1: {x.shape=}')
        previous_output, output = self.left_node(x)
        # print(f'UNetLayer 2: {previous_output.shape=}, {output.shape=}')
        if self.next_layer:
            #Asumimos que los tamaños de los tensores van a ser los mismos del left y next_layer
            next_layer_output = self.next_layer(output)
            # print(f'UNetLayer 3: {next_layer_output.shape=}')
            concatenated_output = torch.cat((previous_output, next_layer_output), dim=1)
            # print(f'UNetLayer 4: {concatenated_output.shape=}')
            _, output = self.right_node(concatenated_output)
        print(f'UNetLayer 5: {output.shape=}, {type(output)=}')
        return output

class UNet(nn.Module):
    """
    Una instancia de UNet representa todo el modelo con las diferentes layers, donde la primera y más "de arriba" es la entrada
    y la salida. El constructor recibe como parametro los canales de una lista de enteros con los in_channels de todas las layers,
    donde el primero representa el in_channels de la entrada al modelo, el segundo es el in_channels del top_layer, luego el 
    in_channels, de la siguiente layer, hasta que el último representa la cantidad de in_channels de la base de la U-Net.
    """
    def __init__(self, input_channel, out_channel, layer_channels):
        super(UNet, self).__init__()
        #Construimos las layers de abajo hacia arriba
        bottom_top_layers = list(reversed(layer_channels))
        layers_pairs = [(bottom_top_layers[i], bottom_top_layers[i+1]) for i in range(len(bottom_top_layers)-1)] + [(bottom_top_layers[-1], bottom_top_layers[-1])]
        # Esto convierte [64, 128, 256, 512, 1024] a [(1024,512), (512,256), (256, 128), (128, 64), (64, 64)]
        print(f'{layers_pairs=}')

        layer_counter = 0
        previous_layer = None
        for (current_channels, previous_channels) in layers_pairs:
            if layer_counter == 0: #Estamos en la base de la U-Net
                left_node = UNetNode(
                    in_channels = previous_channels,
                    out_channels = current_channels,
                    last_operation = nn.ConvTranspose2d(current_channels, previous_channels, kernel_size=2, stride=2)
                )
                right_node = None
            elif layer_counter == len(layers_pairs) - 1: #Estamos en la top layer
                left_node = UNetNode(
                    in_channels = input_channel,
                    out_channels = current_channels,
                    last_operation = nn.MaxPool2d(kernel_size=2, stride=2)
                )
                right_node = UNetNode(
                    in_channels = current_channels*2,
                    out_channels = current_channels,
                    last_operation = nn.Conv2d(current_channels, out_channel, kernel_size=1),
                    top_layer = True
                )
            else:
                left_node = UNetNode(
                    in_channels = previous_channels, 
                    out_channels = current_channels, 
                    last_operation = nn.MaxPool2d(kernel_size=2, stride=2)
                )
                right_node = UNetNode(
                    in_channels = current_channels*2,
                    out_channels = current_channels,
                    last_operation = nn.ConvTranspose2d(current_channels, previous_channels, kernel_size=2, stride=2)
                )
            previous_layer = UNetLayer(left_node, right_node, previous_layer)
            layer_counter += 1
        self.top_layer = previous_layer
        

    def forward(self, x):
        return self.top_layer(x)