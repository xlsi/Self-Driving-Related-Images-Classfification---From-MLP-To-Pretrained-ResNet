import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from glob import glob


os.chdir("../../data")


def get_size(path):
    """This func is used by plot_image_before_tune() and plot_image_after_tune()
    to get image size

    Args:
        path (str): path of image.
    """
    image = cv2.imread(path)
    shape = image.shape[:2]

    return shape[1] * shape[0], shape[1] / shape[0]


def plot_image_before_tune(all_species):
    """Plot image Area and Ratio Distribution before tuning

    Args:
        all_species (list[str]): list of image classes
    """
    # Image Area and Ratio Distribution before tuning
    sizes_origin = []
    ratios_origin = []
    fig_origin, (ax0_origin, ax1_origin) = plt.subplots(1, 2, figsize=(15, 5))
    for i, species in enumerate(all_species):
        paths = sorted(glob(f"Flickr_scrape/{species}/*.*"))
        output = np.array([get_size(path) for path in paths])

        sizes_origin.append(output[:, 0])
        ratios_origin.append(output[:, 1])

        sns.kdeplot(output[:, 0], label=species, ax=ax0_origin)
        sns.kdeplot(output[:, 1], label=species, ax=ax1_origin)
    fig_origin.suptitle("Image Area and Ratio Distribution before tuning", fontsize=16)
    ax0_origin.set_title("Area")
    ax1_origin.set_title("Aspect ratio")
    ax0_origin.legend()
    ax1_origin.legend()
    plt.show()


def plot_image_after_tune(all_species):
    # Image Area and Ratio Distribution after tuning
    sizes_tuned = []
    ratios_tuned = []
    fig_tuned, (ax0_tuned, ax1_tuned) = plt.subplots(1, 2, figsize=(15, 5))
    for i, species in enumerate(all_species):
        paths = sorted(glob(f"Flickr_scrape/{species}/*.*"))
        output = []
        for path in paths:
            res = get_size(path)
            if res[0] >= 400000 and res[0] <= 500000 and res[1] >= 1 and res[1] <= 2:
                output.append(res)
        output = np.array(output)

        sizes_tuned.append(output[:, 0])
        ratios_tuned.append(output[:, 1])

        sns.kdeplot(output[:, 0], label=species, ax=ax0_tuned)
        sns.kdeplot(output[:, 1], label=species, ax=ax1_tuned)
    fig_tuned.suptitle("Image Area and Ratio Distribution after tuning", fontsize=16)
    ax0_tuned.set_title("Area")
    ax1_tuned.set_title("Aspect ratio")
    ax0_tuned.legend()
    ax1_tuned.legend()
    plt.show()


def delete_outlier_image(all_species):
    delete_count = 0
    left_count = 0
    automobile = 0
    automobile_del = 0
    pedestrian = 0
    pedestrian_del = 0
    bicycle = 0
    bicycle_del = 0

    for i, species in enumerate(all_species):
        paths = sorted(glob(f"Flickr_scrape/{species}/*.*"))
        for path in paths:
            res = get_size(path)
            if res[0] < 400000 or res[0] > 500000 or res[1] < 1 and res[1] > 2:
                delete_count += 1
                if species == "automobile":
                    automobile_del += 1
                elif species == "pedestrian":
                    pedestrian_del += 1
                else:
                    bicycle_del += 1
                os.remove(path)
            else:
                left_count += 1
                if species == "automobile":
                    automobile += 1
                elif species == "pedestrian":
                    pedestrian += 1
                else:
                    bicycle += 1
    print(f"Delete {delete_count} pictures in total.")
    print(
        f"Delete {automobile_del} automobile pictures, {pedestrian_del} pedestrain pictures, and {bicycle_del} bicycle pictures."
    )
    print(
        f"There are {automobile} automobile pictures, {pedestrian} pedestrain pictures, and {bicycle} bicycle pictures left as datapoints."
    )
    print(f"There are {left_count} datapoints in total.")


def main():
    all_species = ["automobile", "pedestrian", "bicycle"]
    plot_image_before_tune(all_species)
    plot_image_after_tune(all_species)
    delete_outlier_image(all_species)


if __name__ == "__main__":
    main()

