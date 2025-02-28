import torch
import yaml
import metrics

import torch.utils.data as data
import torchvision.transforms as transforms

from models import LaneNet
from dataset import LaneDataset


with open('configs/eval.yaml', 'r') as configs:
    configs = yaml.load(configs, Loader=yaml.FullLoader)


image_transform = transforms.Compose([
    transforms.Resize((224, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

model = LaneNet()
model = model.to(device)

tp_count = 0
fp_count = 0
fn_count = 0

average_iou = 0.0

print(f'\n---------- Evaluation start at: {str(device).upper()} ----------\n')

with torch.no_grad():
    model.load_state_dict(torch.load(configs['checkpoint-path'], map_location=device, weights_only=True))
    model.eval()

    for index, (images, labels) in enumerate(dataloader, start=1):
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

        average_iou += metrics.iou(predicts.flatten(1, 3), labels.flatten(1, 3)).sum().item()

        print(f'\rProgress: [{index}/{dataloader_size}]', end=' ')

average_iou /= dataset_size

precision = metrics.precision(
    tp_count=tp_count,
    fp_count=fp_count,
)

recall = metrics.recall(
    tp_count=tp_count,
    fn_count=fn_count,
)

f1_score = metrics.f1_score(precision=precision, recall=recall)

print(f'\tprecision: {precision:<8.3f} recall: {recall:<8.3f} f1-score: {f1_score:<8.3f} IoU: {average_iou:.3f}')
print(f'\n---------- Evaluation end ----------\n')
