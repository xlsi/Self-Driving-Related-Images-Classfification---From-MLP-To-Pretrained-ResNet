import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def accuracy(y_pred, y):
    # Compute the accuracy along the rows, averaging along the number of samples
    return np.equal(y_pred, np.array(y)).mean()


def to_categorical(x, class_num):
    # Transform probabilities into categorical predictions row-wise, by simply taking the max probability
    categorical = np.zeros((x.shape[0], class_num))
    res = np.zeros(x.shape[0])
    res[np.arange(x.shape[0])] = x.argmax(axis=1)
    return res


def load_split_train_test(datadir, valid_size=0.2):
    """This func split the initial images randomly to train dataset and validation dataset.
    Hence, no need to manually create folder or move images to build train and validation dataset.

    Args:
        datadir (str): path for image folder
        valid_size (float, optional): ratio of validation dataset. Defaults to 0.2.

    Returns:
        trainloader
        testloader
    """
    transformations = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data = datasets.ImageFolder(datadir, transform=transformations)
    test_data = datasets.ImageFolder(datadir, transform=transformations)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler

    train_idx, test_idx = indices[split:], indices[:split]
    print(
        f"There are {len(train_idx)} training data points and {len(test_idx)} testing data points."
    )
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, batch_size=32
    )
    testloader = torch.utils.data.DataLoader(
        test_data, sampler=test_sampler, batch_size=32
    )
    return trainloader, testloader
