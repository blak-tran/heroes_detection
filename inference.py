import torch
from torchvision import transforms
from PIL import Image
import cv2 
import numpy as np 
import os, glob
from model import SimpleCNN

original_dict = {
    "Ahri": 0,
    "Akali": 1,
    "Alistar": 2,
    "Amumu": 3,
    "Annie": 4,
    "Ashe": 5,
    "Blitzcrank": 6,
    "Braum": 7,
    "Camille": 8,
    "Corki": 9,
    "Darius": 10,
    "Diana": 11,
    "Draven": 12,
    "Evelynn": 13,
    "Ezreal": 14,
    "Fiora": 15,
    "Fizz": 16,
    "Galio": 17,
    "Garen": 18,
    "Gragas": 19,
    "Graves": 20,
    "Janna": 21,
    "Jax": 22,
    "Jhin": 23,
    "Jinx": 24,
    "Katarina": 25,
    "Kennen": 26,
    "Leona": 27,
    "Lulu": 28,
    "Lux": 29,
    "Malphite": 30,
    "Nami": 31,
    "Nasus": 32,
    "Olaf": 33,
    "Orianna": 34,
    "Pantheon": 35,
    "Rakan": 36,
    "Rammus": 37,
    "Rengar": 38,
    "Seraphine": 39,
    "Shyvana": 40,
    "Singed": 41,
    "Sona": 42,
    "Soraka": 43,
    "Teemo": 44,
    "Tristana": 45,
    "Tryndamere": 46,
    "Varus": 47,
    "Vayne": 48,
    "Vi": 49,
    "Wukong": 50,
    "Yasuo": 51,
    "Zed": 52,
    "Ziggs": 53
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((42, 42)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])


# Define a function for inference
def predict_hero(image_path, model):

    # Open and preprocess the input image
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device) # Add batch dimension

    # Perform inference
    with torch.no_grad():
        # Get model predictions
        predictions = model(input_image)

    # Get the predicted class index
    predicted_class_idx = torch.argmax(predictions, dim=1).item()

    # Map the class index to the corresponding hero name using the original_dict
    predicted_hero_name = None
    for hero_name, idx in original_dict.items():
        if idx == predicted_class_idx:
            predicted_hero_name = hero_name
            break

    if predicted_hero_name is not None:
        return predicted_hero_name
    else:
        return "Unknown"

def detect_circles(img_path):
    # Convert to grayscale.
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    height, width, _ = img.shape
    img = img[:, :width // 2]
    # Convert to grayscale. 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
      
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 
      
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                   param2 = 30, minRadius = 1, maxRadius = 40) 
    ret,thresh = cv2.threshold(gray,50,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    print("Number of contours detected:", len(contours))
      
    # Draw circles that are detected. 
    if detected_circles is not None: 
      
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
      
        pt = detected_circles[0, :][0]
        x, y, r = pt[0], pt[1], pt[2]
        cropped_circle = img[y - r:y + r, x - r:x + r]
    elif contours is not None:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w) / h
        
            if ratio >= 0.9 and ratio <= 1.1:
                # Square contour
                square_crop = img[y:y + h, x:x + w]
                return square_crop
            else:
                # Rectangle contour
                rectangle_crop = img[y:y + h, x:x + w]
                return rectangle_crop
                
    return cropped_circle

# Example usage
if __name__ == "__main__":
    # Load the trained model and original_dict (same dictionary used during training)
    model_path = '/home/dattran/datadrive/research/heros_detection/checkpoint_epoch_50.pth'  # Replace with the actual path to your trained model file
    # Load the checkpoint
    checkpoint = torch.load(model_path)

    # Create an instance of the model
    model = SimpleCNN(image_height=42, image_width=42, num_classes=54)

    # Load the model's state dictionary from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()
    
    model = model.to(device)

    img_test_path = "/home/dattran/datadrive/research/heros_detection/datasets/heroes/test_images"
    temp_path = "/home/dattran/datadrive/research/heros_detection/temp"

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        
    imgs_list = sorted(glob.glob(os.path.join(img_test_path, "*.jpg")))
    with open("predicted_heroes.txt", "w") as output_file:
        # Iterate through images
        for idx, img_path in enumerate(imgs_list):
            base_name = os.path.basename(img_path)
            
            # Detect circles or retangle then save cropped image
            cropped_img = detect_circles(img_path)
            img_crop_path = f"{temp_path}/img_{idx}.png"
            cv2.imwrite(img_crop_path, cropped_img)
            
            # Predict the hero
            predicted_hero = predict_hero(img_crop_path, model, original_dict)
            
            # Print the predicted hero name and write it to the file
            print(f"{base_name} Predicted Hero: {predicted_hero}")
            output_file.write(f"{base_name} {predicted_hero}\n")