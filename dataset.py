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
        image = self.images[index]
        label = self.labels[index]

        return self.convert_image(image), torch.tensor(label, dtype=torch.int64)

    def load_annotations(self):
        with open(f'{self.root}/annotations.json', 'r') as annotations:
            return annotations.readlines()

    def setup(self):
        annotations = self.load_annotations()

        for annotation in annotations:
            sample = json.loads(annotation)

            image = sample['image']
            label = sample['label']

            self.images.append(image)
            self.labels.append(label)

    def convert_image(self, name):
        return self.transform(Image.open(f'{self.root}/images/{name}'))
