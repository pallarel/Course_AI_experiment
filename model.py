import torch
import torch.nn as nn
import math
import torchvision


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == 1:
            nn.init.constant_(m.weight, 1)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):  
        nn.init.constant_(m.weight, 1)  
        nn.init.constant_(m.bias, 0)
        
class SimpleCNNPredictor(nn.Module):
    def __init__(self, input_size, input_channel, output_dim):
        assert input_size > 32
        super().__init__()
        
        current_channel = 2**(round(math.log2(input_size)) - 4)     # [256, 512) => 16 channels
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=current_channel, kernel_size=7, padding=3),
            nn.BatchNorm2d(current_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )  # divide factor: 2

        current_size = input_size // 2
        self.network = nn.ModuleList()
        for i in range(round(math.log2(current_size))-3, 0, -1):
            self.network.append(nn.Conv2d(in_channels=current_channel, out_channels=current_channel*2, kernel_size=3, padding=1))
            self.network.append(nn.BatchNorm2d(num_features=current_channel*2))
            self.network.append(nn.ReLU(inplace=True))
            current_channel *= 2
            if i > 1:
                self.network.append(nn.MaxPool2d(kernel_size=2))
                current_size //= 2

        self.last_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=current_channel * (current_size//2)**2, out_features=output_dim)
        )

        self.apply(init_weights)

    def forward(self, x):
        x = self.first_layer(x)
        for i, layer in enumerate(self.network):
            x = layer(x)
        x = self.last_layer(x)
        return x


class EfficientNetPretrained(nn.Module):
    def __init__(self, input_size, input_channel, output_dim):
        assert input_size == 224 and input_channel == 3
        super().__init__()
        
        self.main_model = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT)

        self.last_connection = nn.Sequential(
            nn.Linear(1000, 100), 
            nn.ReLU(inplace=True), 
            nn.Linear(100, output_dim)
        )

        self.last_connection.apply(init_weights)
        
    
    def forward(self, x):
        with torch.no_grad():
            x = self.main_model(x)

        x = self.last_connection(x)
        return x

    #def forward(self, x):
        #x = self.main_model(x)
        #for i, layer in enumerate(self.network):
        #    x = layer(x)
        #x = self.last_layer(x)
        #return x
    


def EfficientNetWrapper(input_size, input_channel, output_dim) -> torchvision.models.EfficientNet:
    assert input_size == 224 and input_channel == 3
    return torchvision.models.efficientnet_b0(num_classes=output_dim)
