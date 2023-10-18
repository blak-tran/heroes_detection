import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import base64
from PIL import Image
from io import BytesIO


# URL of the web page containing the images
url = "https://leagueoflegends.fandom.com/wiki/List_of_champions"

# Specify the path to your text file
file_path = '/home/dattran/datadrive/research/heros_detection/datasets/heroes/hero_names.txt'

heros_list = []
try:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read and print each line in the file
        for line in file:
            hero = line.strip()
            heros_list.append(hero)
            print(line.strip())  # strip() removes the newline character at the end of each line
except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Send an HTTP request to the URL
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, "html.parser")

# Find all img tags in the HTML
img_tags = soup.find_all("img")


save_path_root = "/home/dattran/datadrive/research/heros_detection/datasets/heroes/train_data"
# Extract image URLs and download the images
for img in img_tags:
    img_url = img.get("data-src")  # Get the 'src' attribute of the img tag
    data_image_name = img.get('data-image-name')
    if data_image_name is not None:
        hero_name = data_image_name.split(" ")[0]
        if hero_name in heros_list:
            print("Img Link: ", img_url)
            img_response = requests.get(img_url, stream=True)
            img_path = f"{save_path_root}/{hero_name}.jpg"
            if not os.path.isfile(img_path):
                with open(img_path, "wb") as img_file:
                        for chunk in img_response.iter_content(chunk_size=8192):
                            img_file.write(chunk)
                print(f"Download: {img_path}")
