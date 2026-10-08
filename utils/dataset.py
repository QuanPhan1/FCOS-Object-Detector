import json
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np

class ObjectDetectionDataset(Dataset):
    def __init__(self, json_path, image_dir, target_size=512, is_train=True, multi_scale=False):
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.is_train = is_train
        self.multi_scale = multi_scale
        
        # Multi-scale options
        self.scales = [448, 480, 512, 544, 576, 608, 640] if multi_scale else [target_size]
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.classes = data["classes"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
            
        # Keep all images
        self.images_info = []
        for img in data["images"]:
            img_id = img["id"]
            anns = annotations_by_image.get(img_id, [])
            self.images_info.append({
                "id": img_id,
                "width": img["width"],
                "height": img["height"],
                "annotations": anns
            })
            
    def __len__(self):
        return len(self.images_info)
        
    def _apply_augmentations(self, image, boxes, labels):
        # 1. Random horizontal flip
        if self.is_train and random.random() < 0.5:
            image = ImageOps.mirror(image)
            w, h = image.size
            # boxes: shape [N, 4], formats [xmin, ymin, xmax, ymax]
            flipped_boxes = boxes.copy()
            flipped_boxes[:, 0] = w - boxes[:, 2]
            flipped_boxes[:, 2] = w - boxes[:, 0]
            boxes = flipped_boxes
            
        # 2. Color jittering (only in training)
        if self.is_train and random.random() < 0.5:
            # We can use simple PIL adjustments or skip to save computation
            # Let's apply simple brightness/contrast adjustments via PIL
            from PIL import ImageEnhance
            enhancers = [
                (ImageEnhance.Brightness, 0.2),
                (ImageEnhance.Contrast, 0.2),
                (ImageEnhance.Color, 0.2)
            ]
            random.shuffle(enhancers)
            for Enhancer, factor in enhancers:
                if random.random() < 0.5:
                    enh = Enhancer(image)
                    image = enh.enhance(random.uniform(1.0 - factor, 1.0 + factor))
                    
        # 3. Random safe crop (only in training, with low prob)
        if self.is_train and random.random() < 0.3:
            w, h = image.size
            if len(boxes) > 0:
                # Select a random box as anchor to ensure the crop contains at least this box
                anchor_idx = random.randint(0, len(boxes) - 1)
                anchor_box = boxes[anchor_idx]
                axmin, aymin, axmax, aymax = anchor_box
                
                # Determine safe crop boundaries
                cxmin = int(random.uniform(0, min(axmin, w * 0.2)))
                cymin = int(random.uniform(0, min(aymin, h * 0.2)))
                cxmax = int(random.uniform(max(axmax, w * 0.8), w))
                cymax = int(random.uniform(max(aymax, h * 0.8), h))
                
                image = image.crop((cxmin, cymin, cxmax, cymax))
                new_w, new_h = image.size
                
                # Shift boxes and filter
                shifted_boxes = []
                shifted_labels = []
                for box, label in zip(boxes, labels):
                    xmin, ymin, xmax, ymax = box
                    # Shift coordinates
                    xmin_s = max(0.0, xmin - cxmin)
                    ymin_s = max(0.0, ymin - cymin)
                    xmax_s = min(float(new_w), xmax - cxmin)
                    ymax_s = min(float(new_h), ymax - cymin)
                    
                    # Check if box is still valid and center is inside crop
                    box_w = xmax_s - xmin_s
                    box_h = ymax_s - ymin_s
                    if box_w > 5 and box_h > 5:
                        shifted_boxes.append([xmin_s, ymin_s, xmax_s, ymax_s])
                        shifted_labels.append(label)
                if len(shifted_boxes) > 0:
                    boxes = np.array(shifted_boxes, dtype=np.float32)
                    labels = np.array(shifted_labels, dtype=np.int64)
                    
        return image, boxes, labels

    def __getitem__(self, idx):
        info = self.images_info[idx]
        img_id = info["id"]
        img_path = self.image_dir / img_id
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        # Parse boxes and labels
        boxes = []
        labels = []
        for ann in info["annotations"]:
            # bbox is [xmin, ymin, xmax, ymax]
            boxes.append(ann["bbox"])
            labels.append(self.class_to_idx[ann["class"]])
            
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply augmentations (flips, crops, jitters)
        image, boxes, labels = self._apply_augmentations(image, boxes, labels)
        
        # Always resize to base target_size first (collate_fn will apply multi-scale dynamically)
        target_size = self.target_size
        
        # Resize image
        resized_img = image.resize((target_size, target_size), Image.BILINEAR)
        
        # Scale bounding boxes
        if len(boxes) > 0:
            scale_x = target_size / float(image.width)
            scale_y = target_size / float(image.height)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            # Clip boxes to boundary
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, target_size)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, target_size)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            
        # Convert to tensors
        # Image normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        img_arr = np.array(resized_img, dtype=np.float32) / 255.0
        img_arr = (img_arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float() # [3, H, W]
        
        boxes_tensor = torch.from_numpy(boxes)
        labels_tensor = torch.from_numpy(labels)
        
        return {
            "image": img_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": img_id,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "target_size": torch.tensor([target_size, target_size], dtype=torch.float32),
            "multi_scale": self.multi_scale,
            "scales": self.scales
        }

def collate_fn(batch):
    images = torch.stack([x["image"] for x in batch])
    boxes = [x["boxes"] for x in batch]
    labels = [x["labels"] for x in batch]
    image_ids = [x["image_id"] for x in batch]
    orig_sizes = torch.stack([x["orig_size"] for x in batch])
    target_sizes = torch.stack([x["target_size"] for x in batch])
    
    # Apply multi-scale training at the batch level
    if batch[0].get("multi_scale", False):
        scales = batch[0]["scales"]
        target_size = random.choice(scales)
        base_size = images.shape[-1]
        
        if target_size != base_size:
            # Resize the batch of images dynamically
            images = F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)
            
            # Scale bounding boxes and target sizes accordingly
            scale_factor = target_size / float(base_size)
            boxes = [b * scale_factor for b in boxes]
            target_sizes = torch.full_like(target_sizes, target_size)
            
    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "image_ids": image_ids,
        "orig_sizes": orig_sizes,
        "target_sizes": target_sizes
    }
