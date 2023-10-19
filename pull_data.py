import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import base64
from PIL import Image
from io import BytesIO


# # URL of the web page containing the images https://leagueoflegends.fandom.com/wiki/List_of_champions/Base_statistics
# url = "https://leagueoflegends.fandom.com/wiki/List_of_champions"

# # Specify the path to your text file
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

# # Send an HTTP request to the URL
# response = requests.get(url)

# # Parse the HTML content of the page
# soup = BeautifulSoup(response.content, "html.parser")

# # Find all img tags in the HTML
# img_tags = soup.find_all("img")


save_path_root = "/home/dattran/datadrive/research/heros_detection/datasets/heroes/train_data"
# # Extract image URLs and download the images
# for img in img_tags:
#     img_url = img.get("data-src")  # Get the 'src' attribute of the img tag
#     # data_image_name = img.get('data-image-name')
#     data_image_name = img.get('alt')# Replace spaces with underscores
#     data_image_name = data_image_name.replace(" ", "_")

#     if data_image_name is not None and img_url is not None:
#         hero_name = data_image_name.split(" ")[0]
#         if hero_name in heros_list:
#             print("Img Link: ", img_url)
#             img_response = requests.get(img_url, stream=True)
#             img_path = f"{save_path_root}/{hero_name}.jpg"
#             index = 1
#             while os.path.isfile(img_path):
#                 # If the file exists, try a new filename with an index
#                 img_path = f"{save_path_root}/{hero_name}_{index}.jpg"
#                 index += 1
#             with open(img_path, "wb") as img_file:
#                     for chunk in img_response.iter_content(chunk_size=8192):
#                         img_file.write(chunk)
#             print(f"Download: {img_path}")

import requests

# URL to fetch JSON data
url = "https://champsdb.gg/data/champions.json"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse JSON response
    
    json_data = response.json()
    for data in  json_data:
        # Extract the value of the "name" field
        hero_name = data["name"]
        hero_name = hero_name.replace(" ", "_")
        if hero_name in heros_list:
            url_img =  data["portrait"]
            img_response = requests.get(url_img, stream=True)
            img_path = f"{save_path_root}/{hero_name}.jpg"
            index = 1

            # Check if the original file exists
            while os.path.isfile(img_path):
                # If the file exists, try a new filename with an index
                img_path = f"{save_path_root}/{hero_name}_{index}.jpg"
                index += 1
                
            with open(img_path, "wb") as img_file:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        img_file.write(chunk)
            print(f"Download: {img_path}")
        
else:
    # Print an error message if the request was not successful
    print("Failed to fetch data from the URL")