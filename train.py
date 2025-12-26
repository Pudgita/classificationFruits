import kagglehub
from torchvision import datasets, transforms
import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tfs
from PIL import Image
import torch.utils.data as data
from torchvision import models
from tqdm import tqdm


# Download latest version


class MyDataset(data.Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.dir = os.listdir(self.path)
        self.data = []
        self.targets = []
        self.transform = transform

        for idx, files in enumerate(self.dir):
            for i in os.listdir(os.path.join(path, files)):
                self.data.append(os.path.join(os.path.join(path, files), i))
                self.targets.append(idx)


    def __len__(self):
        return len(self.data)          

    def __getitem__(self, item):
        img_path = self.data[item]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)


        target = self.targets[item]

        return img, target



transforms = tfs.Compose([
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.CenterCrop(224),
])
dataset_train = MyDataset('D:/Kaggle/Fruits/Data/train', transforms)
val_dataset = MyDataset('D:/Kaggle/Fruits/Data/validation', transforms)
test_dataset = MyDataset('D:/Kaggle/Fruits/Data/test', transforms)



train_data = data.DataLoader(dataset_train, batch_size=32, shuffle=True)
data_valid = data.DataLoader(val_dataset, batch_size=16)
data_test = data.DataLoader(test_dataset, batch_size=16)


model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(4096, 36)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizator = optim.Adam(params=model.parameters(), lr=0.001)

epoch = 2
model.train()


for _e in range(epoch):
    tqdm_train = tqdm(train_data, leave=True, colour='blue')
    for x_train, y_train in tqdm_train:
        predict_train = model(x_train) 
        loss_train = loss_func(predict_train, y_train)

        optimizator.zero_grad()
        loss_train.backward()
        optimizator.step()
        tqdm_train.set_description(f'Epoch {_e+1}/{epoch} [Train]')

    model.eval()
    total_val_loss = 0
    batch_count = 0


    val_train = tqdm(data_valid, leave=True, colour='green')
    with torch.no_grad():
        for x_valid, y_valid in val_train:
            predict_val = model(x_valid)
            loss_val = loss_func(predict_val, y_valid)
            total_val_loss += loss_val.item()
            batch_count += 1
            val_train.set_description(f'Epoch {_e+1}/{epoch} [Valid]')

        avg_val_loss = total_val_loss / batch_count
        torch.save(model.state_dict(), 'model.pth')
        
        print(f'Эпоха: {_e + 1}/{epoch}, avg loss = {avg_val_loss:.4f}')

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for x_test, y_test in data_test:
        predict_test = model(x_test)
        predict_label = predict_test.argmax(dim=1)
        test_total += y_test.size(0)
        test_correct += (predict_label == y_test).sum().item()


accuracy = 100.0 * test_correct / test_total
print(f"\n{'='*50}")
print(f"ТЕСТОВЫЕ РЕЗУЛЬТАТЫ:")
print(f"{'='*50}")
print(f"Правильно классифицировано: {test_correct}/{test_total}")
print(f"Точность (Accuracy): {accuracy:.2f}%")
print(f"{'='*50}")