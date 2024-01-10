import torch
import torch.nn as nn
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.CocoDetection() # TODO download dataset
test_dataset = datasets.CocoDetection() # TODO download dataset

