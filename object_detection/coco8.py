import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.CocoDetection() # TODO download dataset
test_dataset = datasets.CocoDetection() # TODO download dataset

class ObjectDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(ObjectDetector, self).__init__()

        self.backbone = torchvision.models.resnet50(pretrained=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes*4)

    def forward(self, x):
        x = self.backbone()

        return x


model = ObjectDetector()

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

