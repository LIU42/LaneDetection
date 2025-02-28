import json
import torch.utils.data as data

from PIL import Image


class LaneDataset(data.Dataset):
    def __init__(self, root, image_transform, label_transform):
        self.root = root
        self.images = []
        self.labels = []
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.setup()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = self.load_image(image)
        label = self.load_label(label)

        return self.image_transform(image), self.label_transform(label).long()

    def load_annotations(self):
        with open(f'{self.root}/annotations.json', 'r') as annotations:
            return annotations.readlines()

    def setup(self):
        annotations = self.load_annotations()

        for annotation in annotations:
            annotation = json.loads(annotation)

            image = annotation['image']
            label = annotation['label']

            self.images.append(image)
            self.labels.append(label)

    def load_image(self, name):
        return Image.open(f'{self.root}/images/{name}').crop((0, 136, 640, 360))

    def load_label(self, name):
        return Image.open(f'{self.root}/labels/{name}').crop((0, 17, 80, 45))
