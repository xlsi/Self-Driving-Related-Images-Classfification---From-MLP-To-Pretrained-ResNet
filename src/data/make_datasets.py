import os
import requests
import pandas as pd
from tqdm import tqdm
import configparser
from flickrapi import FlickrAPI


def define_path_to_file(path_to_file):
    """This func changes the working directory to save images.

    Args:
        path_to_file (str): folder's name that you want to change the images downloaded.
    """

    os.chdir(path_to_file)
    print("This is the path of saving images:", os.getcwd())


def fetch_image_link(query, max_count, key, secret):
    """This funcs is for getting url for images

    Args:
        query (str): query is str type, and stored in a list QUERIES.
        max_count (int): number of images for each class that you want to get.
        key (str): key for Flickr.
        secret (str): secret for Flickr

    Returns:
        urls (list): a list of images' url.
    """
    flickr = FlickrAPI(key, secret)  # initialize python flickr api
    photos = flickr.walk(
        text=query,
        tag_mode="all",
        extras="url_c",  # specify meta data to be fetched
        sort="relevance",
    )  # sort search result based on relevance (high to low by default)

    urls = []
    count = 0

    for photo in photos:
        if count < max_count:
            count = count + 1
            # print("Fetching url for image number {}".format(count))
            try:
                url = photo.get("url_c")
                urls.append(url)
            except:
                print("Url for image number {} could not be fetched".format(count))
        else:
            print(
                f"Done fetching {query} urls, fetched {len(urls)} urls out of {max_count}"
            )
            break
    return urls


def fetch_files_with_link(url_path):
    with open(url_path, newline="") as csvfile:
        urls = pd.read_csv(url_path, delimiter=",")
        urls = urls.iloc[:, 1].to_dict().values()

    SAVE_PATH = os.path.join(url_path.replace("_urls.csv", ""))
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)  # define image storage path

    for idx, url in tqdm(enumerate(urls), total=len(urls)):
        # print("Starting download {} of ".format(url[0] + 1), len(urls))
        try:
            resp = requests.get(url, stream=True)  # request file using url
            path_to_write = os.path.join(SAVE_PATH, url.split("/")[-1])
            outfile = open(path_to_write, "wb")
            outfile.write(resp.content)  # save file content
            outfile.close()
            # print("Done downloading {} of {}".format(idx + 1, len(urls)))
        except:
            print("Failed to download url number {}".format(idx))
    print(f"Done with {url_path} download, images are saved in {SAVE_PATH}")


def main():
    config = configparser.ConfigParser()
    config.read("./config.ini")

    # define the path where you want to save images
    path_to_file = str(config["make_datasets"]["path_to_file"])
    define_path_to_file(path_to_file)

    # getting url for images
    max_count = int(config["make_datasets"]["max_count"])
    key = str(config["make_datasets"]["key"])
    secret = str(config["make_datasets"]["secret"])
    QURIES = ["pedestrian", "automobile", "bicycle"]
    for query in QURIES:
        urls = fetch_image_link(query, max_count, key, secret)
        print("example url:", urls[0])
        urls = pd.Series(urls)
        save_path = "./Flickr_scrape/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        category_path = f"{save_path}/{query}_urls.csv"
        print(f"Writing {query} urls to {category_path}")
        urls.to_csv(category_path)

    # downloading images
    print("Start downloading images...")
    CATEGORIES = ["pedestrian", "automobile", "bicycle"]
    save_path = "./Flickr_scrape/"
    for category in CATEGORIES:
        url_path = f"{save_path}/{category}_urls.csv"
        fetch_files_with_link(url_path)


if __name__ == "__main__":
    main()
