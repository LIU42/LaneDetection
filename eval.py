import torch
import yaml

import torch.utils.data as data
import torchvision.transforms as transforms

from models import LaneNet
from dataset import LaneDataset


with open('configs/eval.yaml', 'r') as configs:
    configs = yaml.safe_load(configs)

transform = transforms.Compose([
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
])

dataset = LaneDataset('datasets/test', transform=transform)
dataset_size = len(dataset)

dataloader = data.DataLoader(dataset, configs['batch-size'], num_workers=configs['num-workers'])
dataloader_size = len(dataloader)

device = torch.device(configs['device'])

model = LaneNet()
model = model.to(device)

delta0_accuracy = 0.0
delta1_accuracy = 0.0
delta2_accuracy = 0.0

print(f'\n---------- Evaluation Start At: {str(device).upper()} ----------\n')\

with torch.no_grad():
    model.load_state_dict(torch.load(configs['model-path'], map_location=device, weights_only=True))
    model.eval()

    for index, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        predicts = torch.argmax(model(images), dim=3)

        delta0_accuracy += (torch.abs(predicts - labels) <= 0).sum().item()
        delta1_accuracy += (torch.abs(predicts - labels) <= 1).sum().item()
        delta2_accuracy += (torch.abs(predicts - labels) <= 2).sum().item()

        print(f'\rEvaluating: [{index}/{dataloader_size}]', end=' ')

    print('\nEvaluating Done!\n')

delta0_accuracy /= 48
delta1_accuracy /= 48
delta2_accuracy /= 48

delta0_accuracy /= dataset_size
delta1_accuracy /= dataset_size
delta2_accuracy /= dataset_size

print(f'Delta0 Accuracy: {delta0_accuracy:.3f}')
print(f'Delta1 Accuracy: {delta1_accuracy:.3f}')
print(f'Delta2 Accuracy: {delta2_accuracy:.3f}')

print(f'\n---------- Evaluation End ----------\n')
