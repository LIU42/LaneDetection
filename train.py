import torch
import yaml

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from models import LaneNet
from dataset import LaneDataset


with open('configs/train.yaml', 'r') as configs:
    configs = yaml.safe_load(configs)

augment_transform = transforms.Compose([
    transforms.Resize((360, 640)),
    transforms.RandomAdjustSharpness(0.1),
    transforms.RandomGrayscale(0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(scale=(0.05, 0.2), ratio=(0.5, 2.0)),
])

normal_transform = transforms.Compose([
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
])

train_transform = normal_transform
valid_transform = normal_transform

if configs['use-augment']:
    train_transform = augment_transform

train_dataset = LaneDataset('datasets/train', train_transform)
valid_dataset = LaneDataset('datasets/valid', valid_transform)

train_loader = data.DataLoader(train_dataset, configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])
valid_loader = data.DataLoader(valid_dataset, configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])

load_path = configs['load-path']
best_path = configs['best-path']
last_path = configs['last-path']

device = torch.device(configs['device'])

model = LaneNet(dropout=configs['dropout'])
model = model.to(device)

if configs['load-checkpoint']:
    model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=configs['learning-rate'], weight_decay=configs['weight-decay'])
scaler = None

if configs['use-amp']:
    scaler = torch.GradScaler()

max_accuracy = 0.0

print(f'\n---------- Training Start at {str(device).upper()} ----------\n')

for epoch in range(configs['epochs']):
    model.train()
    training_loss = 0.0

    for index, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if configs['use-amp']:
            with torch.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs.flatten(0, 2), labels.flatten(0, 2))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs.flatten(0, 2), labels.flatten(0, 2))
            loss.backward()
            optimizer.step()

        training_loss += loss.item()

        print(f'\rBatch Loss: {loss:.5f} [{index}/{len(train_loader)}]', end='')

    model.eval()
    training_loss /= len(train_loader)

    with torch.no_grad():
        valid_accuracy = 0.0

        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            if configs['use-amp']:
                with torch.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)

            valid_accuracy += (torch.argmax(outputs, dim=3) == labels).sum().item()

        valid_accuracy /= 48
        valid_accuracy /= len(valid_dataset)

        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            torch.save(model.state_dict(), best_path)

        torch.save(model.state_dict(), last_path)

    print(f'\tEpoch: {epoch:<6} Loss: {training_loss:<10.5f} Accuracy: {valid_accuracy:<8.3f}')

print('\n---------- Training Finish ----------\n')
print(f'Max Accuracy: {max_accuracy:.3f}')
