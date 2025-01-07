import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import ToPILImage
import dill as pickle
import pandas as pd

def transform_data():
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
        def __init__(self, images_path, bbox_csv_path, label_csv_path, transforms=None):
            self.transforms = transforms
            self.images_path = images_path

            # Load bounding boxes and labels from CSV
            self.bbox_data = pd.read_csv(bbox_csv_path)
            self.label_data = pd.read_csv(label_csv_path)
            
            # List of unique image paths
            self.all_image_paths = sorted(glob.glob(os.path.join(images_path, '*/*.jpg')))
            
            print("Number of images: ", len(self.all_image_paths))

        def load_img(self, img_path):
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Image not found or unable to load: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return image

        def normalize_bbox(self, bboxes, rows, cols):
            norm_bboxes = np.zeros_like(bboxes)
            if cols > 0 and rows > 0:
                norm_bboxes[:, 0] = bboxes[:, 0] / cols
                norm_bboxes[:, 1] = bboxes[:, 1] / rows
                norm_bboxes[:, 2] = bboxes[:, 2] / cols
                norm_bboxes[:, 3] = bboxes[:, 3] / rows
            return norm_bboxes

        def __getitem__(self, index):
            img_path = self.all_image_paths[index]
            print("img_path:",img_path)
            patient_id = os.path.basename(os.path.dirname(img_path))
            print("patient_id:",patient_id)
            unique_id = os.path.splitext(os.path.basename(img_path))[0]
            print("unique_id:",unique_id)

            image = self.load_img(img_path)

            # Get bounding boxes and labels for the current image
            bboxes = self.bbox_data[(self.bbox_data['StudyInstanceUID'] == patient_id) & 
                                    (self.bbox_data['slice_number'] == unique_id)]
            #print("bboxes:",bboxes)
            labels = self.label_data[self.label_data['StudyInstanceUID'] == patient_id]['patient_overall'].values
            print("lllabels:",labels)
            boxes = bboxes[['x', 'y', 'width', 'height']].values if not bboxes.empty else np.empty((0, 4))
            area = (boxes[:, 2] * boxes[:, 3]) if boxes.size > 0 else np.array([0], dtype=np.float32)
            print("boxes:",boxes)
            labels = torch.tensor(labels, dtype=torch.long) if labels.size > 0 else torch.tensor([], dtype=torch.long)
            print("labels:",labels)
            image = (image * 255).astype(np.uint8)
            image_pil = ToPILImage()(image)

            if self.transforms:
                image = self.transforms(image_pil)

            _, h, w = image.shape
            norm_boxes = self.normalize_bbox(boxes, rows=h, cols=w)
            print("norm_boxes:",norm_boxes)
            valid_indices = (norm_boxes[:, 2] > 0) & (norm_boxes[:, 3] > 0)
            valid_boxes = norm_boxes[valid_indices]

            target = {
                'boxes': torch.as_tensor(valid_boxes, dtype=torch.float32) if valid_boxes.size > 0 else torch.empty((0, 4), dtype=torch.float32),
                'labels': labels,
                'image_id': torch.tensor([index]),
                'area': torch.tensor(area[valid_indices], dtype=torch.float32) if valid_indices.any() else torch.tensor([0], dtype=torch.float32)
            }

            return image, target

        def __len__(self):
            return len(self.all_image_pathsb)

    IMAGE_WIDTH = 800    
    IMAGE_HEIGHT = 680

    train_dataset = CustomDataset(
        images_path=os.path.join(os.getcwd(), "converted_images"),
        bbox_csv_path=os.path.join(os.getcwd(), "train_bounding_boxes.csv"),
        label_csv_path=os.path.join(os.getcwd(), "train.csv"),
        transforms=get_train_transform()
    )
    print("Train Dataset: ", train_dataset)

    valid_dataset = CustomDataset(
        images_path=os.path.join(os.getcwd(), "converted_images"),
        bbox_csv_path=os.path.join(os.getcwd(), "train_bounding_boxes.csv"),
        label_csv_path=os.path.join(os.getcwd(), "train.csv"),
        transforms=get_valid_transform()
    )
    print("Valid Dataset: ", valid_dataset)

    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)
#    for j in range(len(train_dataset)):
#        i, a = train_dataset[j]
#        print("Image: ", i)
#        print("Annotations: ", a)
    i, a = train_dataset[15000]
    print("Image: ", i)
    print("Annotations: ", a)
    
    return train_dataset

transform_data()
