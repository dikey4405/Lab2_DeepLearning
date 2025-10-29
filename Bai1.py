from dataset.mnist_dataset import MnistDataset, collate_fn
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from model.lenet import LeNet

def evaluate(dataloader: DataLoader, model = nn.Module) -> dict:
    model.eval()
    predictions = []
    trues = []
    for items in dataloader:
        image: torch.Tensor = items["image"].to("cpu")
        label: torch.Tensor = items["label"].to("cpu")
        output: torch.Tensor = model(image)
        output = torch.argmax(output, dim=-1)
        predictions.extend(output.tolist())
        trues.extend(label.tolist())

    return {
        "precision": precision_score(trues, predictions, average="macro", zero_division=0),
        "recall": recall_score(trues, predictions, average="macro", zero_division=0),
        "f1": f1_score(trues, predictions, average="macro", zero_division=0)
        }

if __name__ == "__main__":

    train_dataset = MnistDataset(
    image_path="/kaggle/input/mnist_for_lab2/Mnist/train-images-idx3-ubyte",
    label_path="/kaggle/input/mnist_for_lab2/Mnist/train-labels-idx1-ubyte"
)

    test_dataset = MnistDataset(
        image_path="/kaggle/input/mnist_for_lab2/Mnist/t10k-images-idx3-ubyte",
        label_path="/kaggle/input/mnist_for_lab2/Mnist/t10k-labels-idx1-ubyte" 
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

    model = LeNet().to("cpu")
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
            images: torch.Tensor = items["image"].to("cpu")
            labels: torch.Tensor = items["label"].to("cpu")

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


       
