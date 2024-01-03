import cv2
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding = 4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root = "./data",
    train = True,
    transform = transform,
    download = False
)
test_dataset = datasets.CIFAR10(
    root = "./data",
    train = False,
    transform = transform,
    download = True
)


batch_size = 64

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)
test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False
)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels = 3,
            out_channels = 32,
            kernel_size = 3,
            padding = 1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 1,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1,
        )
        self.fc1 = torch.nn.Linear(
            in_features = 16 * 16 * 128,
            out_features = 512
        )
        self.fc2 = torch.nn.Linear(
            in_features = 512,
            out_features = 128
        )
        self.fc3 = torch.nn.Linear(
            in_features = 128,
            out_features = 10
        )