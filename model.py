from torch import nn
class SimpleCNN(nn.Module):
    def __init__(self, image_height=64, image_width=64, num_classes=54):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Calculate the total number of output features from the convolutional layers
        total_output_features = 128 * (image_height // 8) * (image_width // 8)  # Three max-pooling layers reduce dimensions by half three times
        self.fc_layer = nn.Sequential(
            nn.Linear(total_output_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer with 50% probability
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = self.fc_layer(x)
        return x