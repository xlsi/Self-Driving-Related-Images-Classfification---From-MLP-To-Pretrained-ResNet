from matplotlib import pyplot as plt


def plot_loss_and_accuracy(train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls):
    # Plot the loss and accuracy curves
    fig_mlp, ax_mlp = plt.subplots(1, 2, figsize=(15, 5))
    fig_mlp.suptitle("MLP Loss and Accuracy Curves", fontsize=16)
    ax_mlp[0].plot(train_loss_ls, label="Train loss")
    ax_mlp[0].plot(test_loss_ls, label="Test loss")
    ax_mlp[0].legend()
    ax_mlp[0].set_xlabel("Epoch")
    ax_mlp[0].set_ylabel("Loss")

    ax_mlp[1].plot(train_acc_ls, label="Train acc")
    ax_mlp[1].plot(test_acc_ls, label="Test acc")
    ax_mlp[1].legend()
    ax_mlp[1].set_xlabel("Epoch")
    ax_mlp[1].set_ylabel("Accuracy")
