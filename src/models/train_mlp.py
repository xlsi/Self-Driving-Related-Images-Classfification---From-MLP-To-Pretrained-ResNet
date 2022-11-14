from utils import accuracy, to_categorical, load_split_train_test
from src.visualization.visualize import plot_loss_and_accuracy
import time
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multilayer Perceptron.
  """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224 * 3, 30),
            nn.Dropout(0.1),  # Applying dropout to avoid over-fitting
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(10, 3),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


def train_model(all_categories, train_loader, val_loader):
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    # Record statistics values
    train_loss_ls = []
    train_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []
    start = time.time()
    # Run the training loop
    print(f"Start training")
    print("-------------------------------------------------------")
    for epoch in range(0, 20):  # 20 epochs at maximum
        mlp.train()

        # Set current loss value
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader):

            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)
            # print(outputs)
            # print(targets)

            # Compute loss
            loss = loss_function(outputs, targets)
            train_loss += loss.item()
            train_acc += accuracy(to_categorical(outputs, len(all_categories)), targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        # Evaluate the Model
        mlp.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = mlp(inputs)
                loss = loss_function(outputs, labels)
                test_loss += loss.item()
                test_acc += accuracy(
                    to_categorical(outputs, len(all_categories)), labels
                )
            test_loss = test_loss / len(val_loader)
            test_acc = test_acc / len(val_loader)
        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_acc)
        test_loss_ls.append(test_loss)
        test_acc_ls.append(test_acc)
        print(
            "After epoch %2d, train_loss | test_loss | train_acc | test_acc: %.3f | %.3f | %.3f | %.3f"
            % (epoch + 1, train_loss, test_loss, train_acc, test_acc)
        )
    # Process is complete.
    tot_time = round((time.time() - start) / 60, 3)
    print(f"Training process has finished. Total time cost is {tot_time} mins.")

    return mlp, train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls


def main():
    # build train loader and val loader
    data_dir = "../../data/Flickr_scrape"
    train_loader, val_loader = load_split_train_test(data_dir, 0.2)
    print(f"Labels: {train_loader.dataset.classes}")

    # train
    all_categories = ["automobile", "pedestrian", "bicycle"]
    mlp, train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls = train_model(
        all_categories, train_loader, val_loader
    )
    plot_loss_and_accuracy(train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls)
    torch.save(mlp, "../../models/MLP.pth")


if __name__ == "__main__":
    main()
