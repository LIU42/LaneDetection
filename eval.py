import torch
import tqdm
import yaml
import metrics

import torch.utils.data as data
import torchvision.transforms as transforms

from models import LaneNet
from dataset import LaneDataset


with open('configs/eval.yaml', 'r') as configs:
    configs = yaml.load(configs, Loader=yaml.SafeLoader)


def format(name, value):
    print(f'{name:<12} {value:.4f}')


image_transform = transforms.Compose([
    transforms.Resize((224, 640)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((28, 80)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
])

dataset = LaneDataset('datasets/test', image_transform=image_transform, label_transform=mask_transform)
dataset_size = len(dataset)

dataloader = data.DataLoader(dataset, batch_size=configs['batch-size'], shuffle=False, num_workers=configs['num-workers'])
dataloader_size = len(dataloader)

device = torch.device(configs['device'])

model = LaneNet(pretrained=False)
model = model.to(device)

print(f'\n---------- Evaluation start at: {str(device).upper()} ----------\n')

with torch.no_grad():
    tp_count = 0
    fp_count = 0
    fn_count = 0
    total_iou = 0.0

    model.load_state_dict(torch.load(configs['checkpoint-path'], map_location=device, weights_only=True))
    model.eval()

    for images, labels in tqdm.tqdm(dataloader, desc='Inference progress', ncols=80):
        images = images.to(device)
        labels = labels.to(device)

        if configs['use-amp']:
            with torch.autocast('cuda'):
                outputs = model(images)
        else:
            outputs = model(images)

        predicts = (outputs > 0).long()

        tp_count += ((predicts == 1) & (labels == 1)).sum().item()
        fp_count += ((predicts == 1) & (labels == 0)).sum().item()
        fn_count += ((predicts == 0) & (labels == 1)).sum().item()

        total_iou += metrics.iou(predicts.flatten(1, 3), labels.flatten(1, 3)).sum().item()

    precision = metrics.precision(tp_count, fp_count)
    format('Precision', precision)

    recall = metrics.recall(tp_count, fn_count)
    format('Recall', recall)

    f1_score = metrics.f1_score(precision, recall)
    format('F1-score', f1_score)

    average_iou = total_iou / dataset_size
    format('IoU', average_iou)

print(f'\n---------- Evaluation finished ----------\n')
