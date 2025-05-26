import torch
from torchvision import transforms
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from dataset import train_dataset, test_dataset
from utils import ZCAWhitening, init_weights

DATA_PATH = "./05/data/"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 50
VAL_SIZE = 3000
BATCH_SIZE = 64

#学習データ
x_train = np.load(DATA_PATH + "x_train.npy")
t_train = np.load(DATA_PATH + "t_train.npy")

#テストデータ
x_test = np.load(DATA_PATH + "x_test.npy")

trainval_data = train_dataset(x_train, t_train)
test_data = test_dataset(x_test)

zca = ZCAWhitening()
zca.fit(trainval_data)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    zca
])

trainval_data = train_dataset(x_train, t_train, transform=train_transform)
train_data, val_data = torch.utils.data.random_split(trainval_data, [len(trainval_data) - VAL_SIZE, VAL_SIZE])

dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_valid = torch.utils.data.DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

NUM_CLASSES = len(np.unique(t_train))
model = models.resnet50(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(dataloader_train):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # 予測結果の取得（最大値のインデックス）
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    model.eval()
    validation_loss = 0.0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():
        for images, labels in dataloader_valid:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    valid_accuracy = 100 * correct_valid / total_valid

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
          f"Train Loss: {train_loss/len(dataloader_train):.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, "
          f"Validation Loss: {validation_loss/len(dataloader_valid):.4f}, "
          f"Validation Acc: {valid_accuracy:.2f}%")


# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in dataloader_test:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")
