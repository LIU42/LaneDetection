import torch
import toml
import tqdm

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from models import LaneDetectionModel
from dataset import LaneDetectionDataset


def calculate_iou(sequence1, sequence2, eps=1e-6):
    intersection_score = (sequence1 * sequence2).sum()

    sequence1_score = sequence1.sum()
    sequence2_score = sequence2.sum()

    return (intersection_score + eps) / (sequence1_score + sequence2_score - intersection_score + eps)


def batched_dice(sequence1, sequence2, eps=1e-6):
    intersection_score = (sequence1 * sequence2).sum(dim=1)

    sequence1_score = sequence1.sum(dim=1)
    sequence2_score = sequence2.sum(dim=1)

    return (2 * intersection_score + eps) / (sequence1_score + sequence2_score + eps)


def criterion(outputs, targets):
    outputs = outputs.flatten(1)
    targets = targets.flatten(1)

    return nn.functional.binary_cross_entropy(outputs, targets) + 1 - batched_dice(outputs, targets).mean()


configs = toml.load('configs/config.toml')

train_transform = transforms.Compose([
    transforms.RandomAdjustSharpness(0.1, 0.3),
    transforms.RandomAutocontrast(0.3),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomErasing(0.4, (0.05, 0.2), (0.5, 2.0)),
])

valid_transform = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

num_epochs = configs['num-epochs']

train_dataset = LaneDetectionDataset('datasets/train', transform=train_transform)
valid_dataset = LaneDetectionDataset('datasets/valid', transform=valid_transform)

train_dataset_size = len(train_dataset)
valid_dataset_size = len(valid_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=True)

train_dataloader_size = len(train_dataloader)
valid_dataloader_size = len(valid_dataloader)

best_iou_score = 0.0
last_iou_score = 0.0

model = LaneDetectionModel(pretrained=configs['load-pretrained'])
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=configs['learning-rate'], weight_decay=configs['weight-decay'])

load_checkpoint_path = configs['load-checkpoint-path']
best_checkpoint_path = configs['best-checkpoint-path']
last_checkpoint_path = configs['last-checkpoint-path']

if configs['load-checkpoint']:
    model.load_state_dict(torch.load(load_checkpoint_path, map_location='cuda', weights_only=True))

print(f'\n---------- training start ----------\n')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm.tqdm(train_dataloader, ncols=80):
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.sigmoid(), labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    train_loss /= train_dataloader_size

    with torch.no_grad():
        all_scores = torch.zeros(0).cuda()
        all_labels = torch.zeros(0).cuda()

        for images, labels in tqdm.tqdm(valid_dataloader, ncols=80):
            images = images.cuda()
            labels = labels.cuda()

            scores = model(images)
            scores = scores.sigmoid()

            scores = scores.flatten()
            labels = labels.flatten()

            all_scores = torch.cat([all_scores, scores])
            all_labels = torch.cat([all_labels, labels])

        all_scores = all_scores.cpu()
        all_labels = all_labels.cpu()

        iou_score = calculate_iou((all_scores > 0.5).int(), all_labels).item()

        if iou_score > best_iou_score:
            best_iou_score = iou_score
            torch.save(model.state_dict(), best_checkpoint_path)

        last_iou_score = iou_score
        torch.save(model.state_dict(), last_checkpoint_path)

    print(f'\nepoch: {epoch + 1}/{num_epochs:<6} loss: {train_loss:<10.5f} IoU: {iou_score:<8.3f}\n')

print(f'best IoU: {best_iou_score:.3f}')
print(f'last IoU: {last_iou_score:.3f}')

print('\n---------- training finished ----------\n')
