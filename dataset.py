import json
import torch
import torch.utils.data as data

from PIL import Image


class LaneDataset(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.images = []
        self.labels = []
        self.transform = transform
        self.setup()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        image = self.open(image)

        return self.transform(image), label

    def load_annotations(self):
        with open(f'{self.root}/annotations.json', 'r') as annotations:
            return json.load(annotations)

    def setup(self):
        annotations = self.load_annotations()

        for image, points in annotations.items():
            samples = torch.tensor(points['samples'], dtype=torch.int64)

            self.images.append(image)
            self.labels.append(torch.where(samples > 0, samples // 8, 80))

    def open(self, image_name):
        return Image.open(f'{self.root}/images/{image_name}')
