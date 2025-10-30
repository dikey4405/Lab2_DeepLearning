from mnist_dataset import MnistDataset, collate_fn
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy
from model.lenet import LeNet

def evaluate(dataloader: DataLoader, model = nn.Module) -> dict:
    model.eval()
    predictions = []
    trues = []
    for items in dataloader:
        image: torch.Tensor = items["image"].to("cuda")
        label: torch.Tensor = items["label"].to("cuda")
        output: torch.Tensor = model(image)
        output = torch.argmax(output, dim=-1)
        predictions.extend(output.tolist())
        trues.extend(label.tolist())

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

def evaluate_per_number(preds, labels):
    for number in range(10):
        number_preds = [1 if p == number else 0 for p in preds]
        number_labels = [1 if l == number else 0 for l in labels]

        print(f"\n----- Evaluation Results for Number -----")
        compute_scores(number_preds, number_labels)
        print("F1_Scores for number {number}:", f1_score(number_labels, number_preds, average="macro", zero_division=0))
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:")
        print(np.array2string(cm, separator=", "))

if __name__ == "__main__":

    train_dataset = MnistDataset(
    image_path="/kaggle/input/mnist-for-lab2/Mnist/train-images-idx3-ubyte",
    label_path="/kaggle/input/mnist-for-lab2/Mnist/train-labels-idx1-ubyte"
)

    test_dataset = MnistDataset(
        image_path="/kaggle/input/mnist-for-lab2/Mnist/t10k-images-idx3-ubyte",
        label_path="/kaggle/input/mnist-for-lab2/Mnist/t10k-labels-idx1-ubyte" 
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = LeNet().to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_score = 0
    best_score_name = "f1"
    for epoch in range(num_epochs):
        losses = []
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()

        for items in train_dataloader:
            images: torch.Tensor = items["image"].to("cuda")
            labels: torch.Tensor = items["label"].to("cuda")

           # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append loss for later verbose
            losses.append(loss.item())

        print (f"\t- loss: {np.array(losses).mean()}")

        scores = evaluate(test_dataloader, model)

        for score_name in scores:
            print(f"\t-{score_name}: {scores[score_name]}")
        
        current_score = scores[best_score_name]
        if current_score > best_score:
            best_score = current_score
            torch.save(
                model.state_dict(),
                "checkpoint/lenet/best_model.pth"
            )
    print("---- Final evaluation ----")
    all_preds = []
    all_labels = []
    model.eval()
    for items in test_dataloader:
        image: torch.Tensor = items["image"].to("cuda")
        label: torch.Tensor = items["label"].to("cuda")
        output: torch.Tensor = model(image)
        output = torch.argmax(output, dim=-1)
        all_preds.extend(output.tolist())
        all_labels.extend(label.tolist())
    evaluate_per_number(all_preds, all_labels)


       






