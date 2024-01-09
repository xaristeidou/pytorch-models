import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

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


class Block(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels,
            out_channels,
            stride: int = 1
    ):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride,
            padding = 1,
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    stride = stride,
                    bias = False
                ),
                nn.BatchNorm2d(num_features = out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class Net(torch.nn.Module):
    '''
    Image classification model, class object, used for CIFAR-10 dataset
    '''
    def __init__(
            self,
            block,
            layers,
            num_classes:int = 10
    ):
        super(Net, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
            self,
            block,
            out_channels,
            blocks,
            stride: int = 1
    ):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def train(
        model: Net,
        num_epochs: int = 30,
        learning_rate: float = 0.001
) -> None:
    '''
    Trains a given model for a specific number of epochs

    -Args:
        model (Net): a PyTorch model based on torch.nn.Module
        epochs (int): number of epochs to train the model
    -Returns:
        (None): Trains and exports a model in .pt format
    '''

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params = model.parameters(),
        lr = learning_rate
    )

    best_accuracy = 0
    for epoch in tqdm(range(num_epochs), desc = "Training process", unit = "Epoch"):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            '''
            Outputs predictions tensor with shape as the output of model's final layer
            and batch_size
            '''
            outputs = model(images)

            '''
            Outputs result of loss function in tensor
            '''
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct = 0
            total = 0

            for images,labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                '''
                We pass the images in tensor with shape of batch_size and then
                we get the outputs in tesnor of batch_size and dimensions of model's
                last layer
                '''
                outputs = model(images)

                '''
                Outputs the position indices (class) for each predicted images
                '''
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "best_model.pt")
        
    print(f"Final Test Accuracy: {accuracy:.2f} %")


def predict(
        image_num: int
) -> None:
    '''
    Run inference and plot an image with prediciton vs true label

    -Args:
        image_num (int): image number index
    -Returns:
        (None): Shows prediction
    '''
    image, label = test_dataset[image_num]
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted_label = torch.argmax(output, dim = 1)

    print(f"True label: {label}")
    print(f"Predicted label: {predicted_label.item()}")

    plt.imshow(test_dataset.data[image_num])
    plt.title(f"Model prediction: {test_dataset.classes[predicted_label.item()]}")
    plt.xlabel(f"True label: {train_dataset.classes[label]}")
    plt.show()


model = Net(Block, [2,2,2,2]).to(device)
train(
    model = model,
    num_epochs = 30,
    learning_rate = 0.001
)
predict(0)