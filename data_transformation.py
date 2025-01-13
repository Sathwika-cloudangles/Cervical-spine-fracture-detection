import os
import glob
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import ToPILImage
import cv2
import numpy as np
import dill as pickle

# Define image dimensions and classes
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
classes = [str(i) for i in range(10)]

def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        T.ToTensor(),
    ])

def get_valid_transform():
    return T.Compose([
        T.ToTensor(),
    ])

class CustomDataset(Dataset):
    def __init__(self, root_path, image_width, image_height, classes, transforms=None):
        self.transforms = transforms
        self.root_path = root_path
        self.image_width = image_width
        self.image_height = image_height
        self.classes = classes
        
        self.all_image_paths = []
        for patient_dir in os.listdir(root_path):
            patient_path = os.path.join(root_path, patient_dir)
            if os.path.isdir(patient_path):
                for img in glob.glob(os.path.join(patient_path, '*.jpg')):
                    self.all_image_paths.append(img)
        
        self.all_image_paths = sorted(self.all_image_paths)
        print(f"Found {len(self.all_image_paths)} images")

    def load_img(self, img_path):
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found or unable to load: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return image

    def __getitem__(self, index):
        img_path = self.all_image_paths[index]
        patient_dir = os.path.dirname(img_path)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        patient_id = os.path.basename(patient_dir)
        
        # Construct annotation paths
        xml_path = os.path.join(patient_dir, "annotations", f"{image_name}.xml")
        txt_path = os.path.join(patient_dir, "labels", f"{image_name}.txt")
        
        # Load image
        image = self.load_img(img_path)
        
        # Initialize variables
        bboxes = []
        labels = []
        xml_label = None
        txt_label = None
        
        # Try XML format first
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Get first object's name as the patient's label
                first_obj = root.find('object')
                if first_obj is not None:
                    xml_label = int(first_obj.find('name').text)
                    print("xml_label----------------------------------------------------------------------------------------------------------------------------",xml_label)
                # Get all bounding boxes
                for obj in root.findall('object'):
                    x = float(obj.find('x').text)
                    y = float(obj.find('y').text)
                    width = float(obj.find('width').text)
                    height = float(obj.find('height').text)
                    
                    x1 = x
                    y1 = y
                    x2 = x + width
                    y2 = y + height
                    
                    label = int(obj.find('name').text)
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(label)
                
                print(f"XML label for {patient_id}: {xml_label}")
                
            except ET.ParseError as e:
                print(f"Error parsing XML file {xml_path}: {e}")
        
        # Try TXT format
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as file:
                    lines = file.readlines()
                if lines:
                    # Get first line's label as the patient's label
                    first_line = lines[0].strip().split()
                    if len(first_line) >= 1:
                        txt_label = int(first_line[0])
                    
                    # Get all bounding boxes
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            label = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            x1 = (x_center - width/2) * self.image_width
                            y1 = (y_center - height/2) * self.image_height
                            x2 = (x_center + width/2) * self.image_width
                            y2 = (y_center + height/2) * self.image_height
                            
                            bboxes.append([x1, y1, x2, y2])
                            labels.append(label)
                
                print(f"TXT label for {patient_id}: {txt_label}")
                
            except Exception as e:
                print(f"Error parsing TXT file {txt_path}: {e}")
        
        # If no bounding boxes were found in either file
        if len(bboxes) == 0:
            print(f"No bounding boxes found for {patient_id}")
            bboxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
            
            # If we have labels from either file, use them
            if xml_label is not None and txt_label is not None:
                if xml_label == txt_label:
                    print(f"Using matching label {xml_label} for {patient_id}")
                    labels = np.array([xml_label], dtype=np.int64)
                else:
                    print(f"Warning: Label mismatch for {patient_id} - XML: {xml_label}, TXT: {txt_label}")
            elif xml_label is not None:
                print(f"Using XML label {xml_label} for {patient_id}")
                labels = np.array([xml_label], dtype=np.int64)
            elif txt_label is not None:
                print(f"Using TXT label {txt_label} for {patient_id}")
                labels = np.array([txt_label], dtype=np.int64)
        else:
            # Convert lists to numpy arrays
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        # Calculate areas
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) if bboxes.size > 0 else np.array([], dtype=np.float32)
        
        # Convert image to PIL and apply transforms
        image = (image * 255).astype(np.uint8)
        image_pil = ToPILImage()(image)
        
        if self.transforms:
            image = self.transforms(image_pil)

        # Normalize bounding boxes
        if bboxes.size > 0:
            bboxes[:, [0, 2]] /= self.image_width
            bboxes[:, [1, 3]] /= self.image_height

        target = {
            'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([index]),
            'area': torch.as_tensor(area, dtype=torch.float32),
            'patient_id': patient_id  # Added patient_id to target
        }

        return image, target

    def __len__(self):
        return len(self.all_image_paths)

def transform_data():
    root_path = os.path.join(os.getcwd(), "converted_images")
    
    train_dataset = CustomDataset(
        root_path=root_path,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        classes=classes,
        transforms=get_train_transform()
    )
    
    valid_dataset = CustomDataset(
        root_path=root_path,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        classes=classes,
        transforms=get_valid_transform()
    )

    print("Testing one sample from training dataset:")
    # for j in range(len(train_dataset)):
    #     i, a = train_dataset[j]
    #     print("Image shape:", i.shape)
    #     print("Annotations:", a)
    i, a = train_dataset[2020]
    print("Image shape:", i.shape)
    print("Annotations:", a)  
    print("Saving datasets to pickle files...")
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)
    
    return train_dataset, valid_dataset

if __name__ == "__main__":
    transform_data()
