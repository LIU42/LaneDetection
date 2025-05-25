import json
import torchvision.io as io

from torch.utils.data import Dataset


class LaneDetectionDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.images = []
        self.labels = []
        self.transform = transform
        self.setup_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = io.read_image(f'{self.root}/images/{self.images[index]}')
        label = io.read_image(f'{self.root}/labels/{self.labels[index]}')

        image = image.float() / 255.0
        label = label.float() / 255.0

        return self.transform(image), label
    
    def setup_dataset(self):
        annotations = self.load_annotations()

        for annotation in annotations:
            annotation = json.loads(annotation.strip())

            image = annotation['image']
            label = annotation['label']

            self.images.append(image)
            self.labels.append(label)

    def load_annotations(self):
        with open(f'{self.root}/annotations.json', 'r') as annotations:
            return annotations.readlines()
