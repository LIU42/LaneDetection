import torch
import yaml
import metrics

import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from models import LaneNet
from dataset import LaneDataset


with open('configs/train.yaml', 'r') as configs:
    configs = yaml.load(configs, Loader=yaml.FullLoader)


def criterion(outputs, labels, eps=1e-6):
    return 1 - metrics.dice(outputs, labels, eps=eps).mean()


augment_transform = transforms.Compose([
    transforms.Resize((224, 640)),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAdjustSharpness(0.1),
    transforms.RandomGrayscale(0.2),
    transforms.RandomErasing(scale=(0.05, 0.2), ratio=(0.5, 2.0)),
])

normal_transform = transforms.Compose([
    transforms.Resize((224, 640)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((28, 80)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

train_transform = normal_transform
valid_transform = normal_transform

if configs['use-augment']:
    train_transform = augment_transform

train_dataset = LaneDataset('datasets/train', image_transform=train_transform, label_transform=mask_transform)
valid_dataset = LaneDataset('datasets/valid', image_transform=valid_transform, label_transform=mask_transform)

train_dataset_size = len(train_dataset)
valid_dataset_size = len(valid_dataset)

train_dataloader = data.DataLoader(train_dataset, batch_size=configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])
valid_dataloader = data.DataLoader(valid_dataset, batch_size=configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])

train_dataloader_size = len(train_dataloader)
valid_dataloader_size = len(valid_dataloader)

load_path = configs['load-path']
best_path = configs['best-path']
last_path = configs['last-path']

device = torch.device(configs['device'])

model = LaneNet(pretrained=configs['load-pretrained'])
model = model.to(device)

if configs['load-checkpoint']:
    model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))

best_average_iou = 0.0
optimizer = optim.Adam(model.parameters(), lr=configs['learning-rate'], weight_decay=configs['weight-decay'])

if configs['use-amp']:
    scaler = torch.GradScaler()

print(f'\n---------- Training Start at {str(device).upper()} ----------\n')

for epoch in range(configs['num-epochs']):
    model.train()
    train_average_loss = 0.0

    for index, (images, labels) in enumerate(train_dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if configs['use-amp']:
            with torch.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs.flatten(1, 3).sigmoid(), labels.flatten(1, 3))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs.flatten(1, 3).sigmoid(), labels.flatten(1, 3))
            loss.backward()
            optimizer.step()

        train_average_loss += loss.item()

        print(f'\rBatch Loss: {loss:.5f} [{index}/{train_dataloader_size}]', end='')

    model.eval()
    train_average_loss /= train_dataloader_size

    with torch.no_grad():
        valid_average_iou = 0.0

        for images, labels in valid_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if configs['use-amp']:
                with torch.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)

            valid_average_iou += metrics.iou((outputs.flatten(1, 3) > 0).long(), labels).sum().item()

        valid_average_iou /= valid_dataset_size

        if valid_average_iou > best_average_iou:
            best_average_iou = valid_average_iou
            torch.save(model.state_dict(), best_path)

        torch.save(model.state_dict(), last_path)

    print(f'\tEpoch: {epoch:<6} Loss: {train_average_loss:<10.5f} IoU: {valid_average_iou:<8.3f}')

print('\n---------- Training Finish ----------\n')

print(f'Best IoU: {best_average_iou:.3f}')
print(f'Last IoU: {valid_average_iou:.3f}')
