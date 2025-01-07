from PIL import Image
import os
import random

def datavisualization():
    # Path to the main images folder
    images_root = "./converted_images"
    
    # Collect all .jpg files from subdirectories
    jpg_files = []
    for root, _, files in os.walk(images_root):
        for file in files:
            if file.endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))

    # Print total images found
    print(f"Total JPG files found: {len(jpg_files)}")

    # Randomly select up to 6 images for visualization
    random_files = random.sample(jpg_files, min(6, len(jpg_files)))
    print("Randomly selected files for visualization:", random_files)
    
    # Visualize and save the selected images to the current working directory
    for file_path in random_files:
        img = Image.open(file_path)
        img_name = os.path.basename(os.path.splitext(file_path)[0])  # Get file name without extension
        output_image_path = os.path.join(os.getcwd(), f"{img_name}.jpg")
        img.save(output_image_path)
        print(f"Image saved for visualization: {output_image_path}")

datavisualization()
