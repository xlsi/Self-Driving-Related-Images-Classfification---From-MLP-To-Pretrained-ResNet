from utils import accuracy, to_categorical, load_split_train_test
from src.visualization.visualize import plot_loss_and_accuracy
import time
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


def train_model(all_categories, train_loader, val_loader):
    # Get CNN model using torchvision.models as models library
    model = models.resnet50(pretrained=False)
    num_labels = 3  # PUT IN THE NUMBER OF LABELS IN YOUR DATA
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, num_labels),
    )
    # Find the device available to use using torch library
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to the device specified above
    model.to(device)
    # Set the error function using torch.nn as nn library
    criterion = nn.CrossEntropyLoss()
    # Set the optimizer function using torch.optim as optim library
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 10
    train_loss_ls = []
    train_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []
    start = time.time()
    # Run the training loop
    print(f"Start training")
    print("-------------------------------------------------------")
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0
        # Training the model
        model.train()

        for inputs, labels in train_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear optimizers
            optimizer.zero_grad()
            # Forward pass
            output = model.forward(inputs)
            # Loss
            loss = criterion(output, labels)
            # Calculate gradients (backpropogation)
            loss.backward()
            # Adjust parameters based on gradients
            optimizer.step()
            # Caclulate loss and accuracy
            step_loss = loss.item()
            step_acc = accuracy(
                to_categorical(output.cpu(), len(all_categories)), labels.cpu()
            )
            train_loss += step_loss
            train_acc += step_acc

        # Evaluating the model
        model.eval()

        # Tell torch not to calculate gradients
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                output = model.forward(inputs)
                # Calculate Loss
                valloss = criterion(output, labels)
                # Caclulate loss and accuracy
                step_loss = valloss.item()
                step_acc = accuracy(
                    to_categorical(output.cpu(), len(all_categories)), labels.cpu()
                )
                val_loss += step_loss
                val_acc += step_acc

        # Get the average loss for the entire epoch
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_acc = train_acc / len(train_loader)
        val_acc = val_acc / len(val_loader)

        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_acc)
        test_loss_ls.append(val_loss)
        test_acc_ls.append(val_acc)
        # Print out the result
        print(
            "After epoch %2d, train_loss | test_loss | train_acc | test_acc: %.3f | %.3f | %.3f | %.3f"
            % (epoch + 1, train_loss, val_loss, train_acc, val_acc)
        )
    tot_time = round((time.time() - start) / 60, 3)
    print(f"Training process has finished. Total time cost is {tot_time} mins.")
    return model, train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls


def main():
    # build train loader and val loader
    data_dir = "../../data/Flickr_scrape"
    train_loader, val_loader = load_split_train_test(data_dir, 0.2)
    print(f"Labels: {train_loader.dataset.classes}")

    # train
    all_categories = ["automobile", "pedestrian", "bicycle"]
    unpre_resnet, train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls = train_model(
        all_categories, train_loader, val_loader
    )
    plot_loss_and_accuracy(train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls)
    torch.save(unpre_resnet, "../../models/Unpretrained ResNet.pth")


if __name__ == "__main__":
    main()
