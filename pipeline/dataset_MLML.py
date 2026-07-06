import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np
from utils.pml_data import generate_uniform_cv_candidate_labels
import os

class DataSet_MLML(Dataset):
    def __init__(self,
                ann_files,
                augs,
                img_size,
                dataset,
                partial_rate,):
        self.dataset = dataset
        self.ann_files = ann_files
        self.augment = self.augs_function(augs, img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ]
        )
        self.anns = []
        self.load_anns()
        print(self.augment)

        self.true_labels = []
        for an in self.anns:
            labels = an['target']
            self.true_labels.append(labels)
        self.true_labels = np.array(self.true_labels)

        self.labels = np.array(self.true_labels).astype(np.int)
        self.labels[self.labels == 0] = -1

        self.changedLabels = self.labels
        self.changedLabels = changeLabelProportion(self.labels, partial_rate)

        Partial_Labels_1 = self.changedLabels.tolist()
        for i in range(self.changedLabels.shape[0]):
            self.anns[i]['target'] = Partial_Labels_1[i]

        self.Partial = torch.Tensor(self.changedLabels)

    def augs_function(self, augs, img_size):            
        t = []
        if 'randomflip' in augs:
            t.append(transforms.RandomHorizontalFlip())
        if 'ColorJitter' in augs:
            t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        if 'resizedcrop' in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if 'RandAugment' in augs:
            t.append(RandAugment())

        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)
    
    def load_anns(self):
        self.anns = []
        for ann_file in self.ann_files:
            json_data = json.load(open(ann_file, "r"))
            self.anns += json_data

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]
        img = Image.open(ann["img_path"]).convert("RGB")

        img = self.augment(img)
        img = self.transform(img)
        message = {
            "img_path": ann["img_path"],
            "target": torch.Tensor(ann["target"]),
            "img": img,
            "index_num": idx
        }

        return message
def changeLabelProportion(labels, label_proportion):

    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape
    return label