import cv2
import os
from roboflow import Roboflow

# Load the Roboflow model (replace 'your_model_id' and 'your_api_key' with your actual model ID and API key)
rf = Roboflow(api_key="TMcBStNa8S94JRFn5YJo")
model = rf.workspace().project("cancer-cell-detection-dkkhm").version("5").model

# Define the directory containing the large images
image_directory = r"D:\Documents\Y3T1\Exploring Robotics & AI\image"

# Define the directory where cropped images will be saved
output_directory = r"D:\Documents\Y3T1\Exploring Robotics & AI\cropped"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize a counter for naming cropped images
image_counter = 1

# Iterate through all images in the image directory
for filename in os.listdir(image_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # You can add other extensions if needed
        image_path = os.path.join(image_directory, filename)

        # Load the image
        image = cv2.imread(image_path)

        # Perform object detection using the model
        predictions = model.predict(image_path).json()

        # Iterate over all detected cancer cells
        for i, prediction in enumerate(predictions["predictions"]):
            # Get the bounding box of each cancer cell (x, y, width, height)
            x1 = int(prediction["x"] - prediction["width"] / 2)
            y1 = int(prediction["y"] - prediction["height"] / 2)
            x2 = int(prediction["x"] + prediction["width"] / 2)
            y2 = int(prediction["y"] + prediction["height"] / 2)

            # Crop the cancer cell from the original image
            cropped_cell = image[y1:y2, x1:x2]

            # Save the cropped cancer cell with a unique number (counter)
            output_path = os.path.join(output_directory, f"{image_counter}.png")
            cv2.imwrite(output_path, cropped_cell)

            print(f"Cropped cancer cell saved as: {output_path}")

            # Increment the counter for the next cropped image
            image_counter += 1

print("Cropping completed for all images.")
