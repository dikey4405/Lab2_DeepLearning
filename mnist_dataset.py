import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np

def collate_fn(items: list[dict]) -> dict:
    images = np.stack([item["image"] for item in items], axis=0).astype(np.float32)
    labels = np.array([item["label"] for item in items], dtype=np.int64)
    return {
        "image": torch.tensor(images, dtype=torch.float32),
        "label": torch.tensor(labels, dtype=torch.long)
    }

class Items:
    def __init__(self, image, label):
        self.image = image
        self.label = label

class MnistDataset(Dataset):
    def __init__(self, image_path: str, label_path: str):
        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)

        self.data = [
            {
                "image": np.array(image, dtype=np.float32),
                "label": label
            }
            for image, label in zip(images.tolist(), labels.tolist())
        ] 

    def __len__(self) -> int:
        return len(self.data)  

    def __getitem__(self, index: int) -> dict:
        image = self.data[index]["image"] / 255.0
        label = self.data[index]["label"]
        return {
            "image": image,
            "label": label
        }