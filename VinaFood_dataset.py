import os
from PIL import Image
import torch
from torch.utils.data import Dataset

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    return {"image": imgs, "label": labels}

class VinaFood(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples, self.classes, self.class_to_idx = self.load_data()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def load_data(self):
        samples = []
        classes = sorted(os.listdir(self.root_dir))
        cls_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                label = cls_to_idx[cls]
                samples.append((path, label))

        return samples, classes, cls_to_idx



