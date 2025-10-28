import torch 
from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np

def collate_fn(samples: list[dict]) -> dict:
    images = [sample["image"].permute(2, 0, 1).unsqueeze(0) for sample in samples]
    labels = [sample["label"] for sample in samples]

    images = torch.cat(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "image": images,
        "label": labels
    }

class VinaFood(Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.labels2idx = {}
        self.idx2labels = {}
        self.data: list[dict] = self.load_data(path)

    def load_data(self, path: str):

        data = []
        label_id = 0

        for folder in os.listdir(path):
            label = folder
            if label not in self.labels2idx:
                self.labels2idx[label] = label_id
                label_id += 1

            for image_file in os.listdir(os.path.join(path, folder)):
                image = cv.imread(os.path.join(path, folder, image_file))
                data.append({
                    "image_path": image,
                    "label": label
                })
        
        self.idx2labels = {id: label for label, id in self.labels2idx.items()}
        return data
    
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        item = self.data[index]
        image = item["image"]
        label = item["label"]

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (224, 224))

        image = torch.tensor(image, dtype=torch.float32) / 255.0
        label_id = self.labels2idx[label]

        return {
            "image": image,
            "label": label_id
        }


