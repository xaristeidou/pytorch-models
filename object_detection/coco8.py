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



    def forward(self): # TODO create the structure when model is ready

        return x