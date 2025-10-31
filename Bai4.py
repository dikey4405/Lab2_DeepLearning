from VinaFood_dataset import VinaFood, collate_fn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from model.pretrained_resnet import PretrainedResnet
import os

def evaluate(dataloader: DataLoader, model: nn.Module, device: torch.device) -> dict:
    model.eval()
    predictions = []
    trues = []
    with torch.no_grad():
        for items in dataloader:
            image: torch.Tensor = items["image"].to(device)
            label: torch.Tensor = items["label"].to(device)
            output: torch.Tensor = model(image)
            output = torch.argmax(output, dim=-1)
            predictions.extend(output.cpu().tolist())
            trues.extend(label.cpu().tolist())

    return {
        "precision": precision_score(trues, predictions, average="macro", zero_division=0),
        "recall": recall_score(trues, predictions, average="macro", zero_division=0),
        "f1": f1_score(trues, predictions, average="macro", zero_division=0)
    }
    
def compute_scores(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }

def evaluate_per_class(preds, labels, num_classes):
    for cls in range(num_classes):
        class_preds = [1 if p == cls else 0 for p in preds]
        class_labels = [1 if l == cls else 0 for l in labels]

        print(f"\n----- Evaluation Results for Class {cls} -----")
        scores = compute_scores(class_labels, class_preds)
        print(f"Accuracy: {scores['accuracy']:.4f}")
        print(f"Precision: {scores['precision']:.4f}")
        print(f"Recall: {scores['recall']:.4f}")
        print(f"F1_Score: {scores['f1']:.4f}")

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Use local dataset paths (adjust if running on Kaggle)
    train_dataset = VinaFood("/kaggle/input/dataset-for-lab2/VinaFood21/train", transform = transform)
    test_dataset = VinaFood("/kaggle/input/dataset-for-lab2/VinaFood21/test", transform = transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model: allow number of classes to be derived from dataset
    num_classes = len(train_dataset.labels_to_names) if hasattr(train_dataset, 'labels_to_names') else 21
    model = PretrainedResnet(num_classes=num_classes).to(device)

    # Loss and optimizer: split lr for backbone and classifier
    loss_fn = nn.CrossEntropyLoss()
    base_lr = 1e-3
    optimizer = optim.Adam([
        {"params": model.resnet.parameters(), "lr": base_lr * 0.1},
        {"params": model.classifier.parameters(), "lr": base_lr},
    ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 10
    best_score = 0
    best_score_name = "f1"
    best_model_path = "best_resnet.pth"

    for epoch in range(num_epochs):
        losses = []
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()

        for items in train_dataloader:
            images: torch.Tensor = items["image"].to(device)
            labels: torch.Tensor = items["label"].to(device)

            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append loss for later verbose
            losses.append(loss.item())

        scheduler.step()

        avg_loss = float(np.array(losses).mean()) if len(losses) > 0 else 0.0
        print(f"\t- loss: {avg_loss:.6f}")

        scores = evaluate(test_dataloader, model, device)

        for score_name in scores:
            print(f"\t-{score_name}: {scores[score_name]:.6f}")

        current_score = scores["f1"]
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), best_model_path)
            print(f"\t Saved best model with F1 = {best_score:.4f}")

    print("\n---- Final Evaluation ----")

    # Load lại model tốt nhất
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for items in test_dataloader:
            images = items["image"].to("cuda")
            labels = items["label"].to("cuda")
            outputs = model(images)
            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    evaluate_per_class(all_preds, all_labels, num_classes=21)


       


