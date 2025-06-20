import torch
import toml

import sklearn.metrics as metrics
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from models import LaneDetectionModel
from dataset import LaneDetectionDataset


def f1_score(scores, labels, threshold):
    return metrics.f1_score(labels, (scores > threshold).int())


def calculate_iou(sequence1, sequence2, eps=1e-6):
    intersection_score = (sequence1 * sequence2).sum()

    sequence1_score = sequence1.sum()
    sequence2_score = sequence2.sum()

    return (intersection_score + eps) / (sequence1_score + sequence2_score - intersection_score + eps)


def iou_score(scores, labels, threshold):
    return calculate_iou((scores > threshold).int(), labels)


configs = toml.load('configs/config.toml')

transform = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = LaneDetectionDataset('datasets/test', transform=transform)
dataset_size = len(dataset)

dataloader = DataLoader(dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=False)
dataloader_size = len(dataloader)

model = LaneDetectionModel(pretrained=False)
model = model.cuda()

log_interval = configs['log-interval']

print(f'\n---------- evaluation start ----------\n')

with torch.no_grad():
    all_scores = []
    all_labels = []

    model.load_state_dict(torch.load(configs['load-checkpoint-path'], map_location='cuda', weights_only=True))
    model.eval()

    for batch, (images, labels) in enumerate(dataloader, start=1):
        images = images.cuda()
        labels = labels.cuda()

        scores = model(images).sigmoid()

        scores = scores.flatten()
        labels = labels.flatten()

        all_scores.append(scores)
        all_labels.append(labels)

        if batch % log_interval == 0:
            print(f'[valid] [{batch:04d}/{dataloader_size:04d}]')

    scores = torch.cat(all_scores).cpu()
    labels = torch.cat(all_labels).cpu()

    auc_score = metrics.roc_auc_score(labels, scores)

    f1_score30 = f1_score(scores, labels, 0.3)
    f1_score40 = f1_score(scores, labels, 0.4)
    f1_score50 = f1_score(scores, labels, 0.5)
    f1_score60 = f1_score(scores, labels, 0.6)
    f1_score70 = f1_score(scores, labels, 0.7)

    ap_score = metrics.average_precision_score(labels, scores)

    iou30 = iou_score(scores, labels, 0.3)
    iou40 = iou_score(scores, labels, 0.4)
    iou50 = iou_score(scores, labels, 0.5)
    iou60 = iou_score(scores, labels, 0.6)
    iou70 = iou_score(scores, labels, 0.7)

    print('\n--------------------------------')
    print(f'F1-Score@30: {f1_score30:.4f}')
    print(f'F1-Score@40: {f1_score40:.4f}')
    print(f'F1-Score@50: {f1_score50:.4f}')
    print(f'F1-Score@60: {f1_score60:.4f}')
    print(f'F1-Score@70: {f1_score70:.4f}')

    print('\n--------------------------------')
    print(f'IoU@30: {iou30:.4f}')
    print(f'IoU@40: {iou40:.4f}')
    print(f'IoU@50: {iou50:.4f}')
    print(f'IoU@60: {iou60:.4f}')
    print(f'IoU@70: {iou70:.4f}')

    print(f'\nAUC: {auc_score:<8.4f} AP: {ap_score:.4f}')

print(f'\n---------- evaluation finished ----------\n')
