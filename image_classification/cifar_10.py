import cv2
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.05)
        x = self.fc3(x)

        return x
    

model = Net().to(device)
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(
    params = model.parameters(),
    lr = learning_rate
)

best_accuracy = 0
num_epochs = 20
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
            torch.save(model.state_dict(), "best_model.pth")