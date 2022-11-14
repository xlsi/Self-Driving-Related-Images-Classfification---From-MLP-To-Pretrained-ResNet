import torch
import torch.autograd.variable as Variable
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def predict_image(image):  # sourcery skip: inline-immediately-returned-variable
    """Predict the class of a specific image

    Args:
        image (_type_): a Pillow image, not a file path
    """
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


def get_random_images(num):
    """Pick a number of random images from the dataset folders

    Args:
        num (int): number of images
    """
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler

    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels


def main():
    # declare the image folder again
    data_dir = "../../data/Flickr_scrape"
    test_transforms = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(),]
    )

    # check for GPU availability, load the model and put it into evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("Pretrained ResNet.pth")
    model.eval()

    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(5)
    fig = plt.figure(figsize=(10, 10))
    for ii in range(len(images)):
        image = to_pil(images[ii])
        index = predict_image(image)
        sub = fig.add_subplot(1, len(images), ii + 1)
        res = int(labels[ii]) == index
        sub.set_title(str(classes[index]) + ":" + str(res))
        plt.axis("off")
        plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()

