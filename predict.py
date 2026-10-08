import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

from models.detector import FCOS
from utils.od_utils import get_fpn_coords, multiclass_nms

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Object Detection Boxes")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to run prediction on")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON file name")
    parser.add_argument("--model_path", type=str, default="./models/best.pth", help="Path to best.pth model weights")
    parser.add_argument("--target_size", type=int, default=512, help="Input size for the model")
    parser.add_argument("--conf_thresh", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.5, help="NMS IoU threshold")
    return parser.parse_args()

def preprocess_image(img_path, target_size=512):
    """
    Load image, resize, normalize and convert to tensor.
    """
    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    
    # Resize
    resized_img = image.resize((target_size, target_size), Image.BILINEAR)
    
    # Normalize
    img_arr = np.array(resized_img, dtype=np.float32) / 255.0
    img_arr = (img_arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float() # [3, H, W]
    
    return img_tensor.unsqueeze(0), orig_w, orig_h # add batch dimension

def main():
    args = parse_args()
    
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: image_dir {image_dir} does not exist.")
        return
        
    output_path = Path(args.output)
    
    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Fallback: check relative to the directory containing predict.py
        alt_path = Path(__file__).resolve().parent / args.model_path
        if alt_path.exists():
            model_path = alt_path
        else:
            print(f"Error: model weights not found at {model_path} or {alt_path}. Please train the model first.")
            return
        
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    classes = checkpoint.get("classes", ["person", "car", "dog", "cat", "chair"])
    num_classes = len(classes)
    
    model = FCOS(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Find all images in the directory
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in valid_exts])
    
    if len(img_paths) == 0:
        print(f"No valid images found in {image_dir}")
        # Return an empty list to satisfy requirements
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return
        
    print(f"Found {len(img_paths)} images. Running predictions...")
    
    # Pre-generate FPN grid coords (since all images are resized to args.target_size)
    coords = get_fpn_coords(height=args.target_size, width=args.target_size, strides=[8, 16, 32], device=device)
    all_x = torch.cat([c["x"] for c in coords]) # [M]
    all_y = torch.cat([c["y"] for c in coords]) # [M]
    all_strides = torch.cat([torch.full_like(c["x"], c["stride"]) for c in coords]) # [M]
    
    predictions_json = []
    
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            image_id = img_path.name # e.g. img_7fd91a4c2e30.jpg
            
            # Preprocess image
            img_tensor, orig_w, orig_h = preprocess_image(img_path, target_size=args.target_size)
            img_tensor = img_tensor.to(device)
            
            # Forward pass
            outputs = model(img_tensor)
            
            cls_logits = outputs["cls_logits"][0] # [M, num_classes]
            reg_preds = outputs["reg_preds"][0] # [M, 4] (l, t, r, b)
            centerness = outputs["centerness"][0].sigmoid() # [M, 1]
            
            # Calculate final detection scores
            cls_probs = torch.sigmoid(cls_logits) # [M, num_classes]
            scores = cls_probs * centerness # [M, num_classes]
            
            # Get max class score and label for each location
            max_scores, labels = torch.max(scores, dim=1) # [M], [M]
            
            # Filter detections by confidence
            keep_mask = max_scores >= args.conf_thresh
            if not keep_mask.any():
                predictions_json.append({
                    "image_id": image_id,
                    "boxes": []
                })
                continue
                
            pos_scores = max_scores[keep_mask]
            pos_labels = labels[keep_mask]
            pos_reg = reg_preds[keep_mask]
            
            pos_x = all_x[keep_mask]
            pos_y = all_y[keep_mask]
            pos_strides = all_strides[keep_mask]
            
            # Convert reg offsets to absolute boxes in model input scale (target_size)
            l = pos_reg[:, 0] * pos_strides
            t = pos_reg[:, 1] * pos_strides
            r = pos_reg[:, 2] * pos_strides
            b = pos_reg[:, 3] * pos_strides
            
            xmin = pos_x - l
            ymin = pos_y - t
            xmax = pos_x + r
            ymax = pos_y + b
            
            boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1) # [num_pos, 4]
            
            # Scale coordinates back to original image size
            scale_x = orig_w / args.target_size
            scale_y = orig_h / args.target_size
            
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            # Clip boxes to original image boundary
            boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, orig_h)
            
            # Apply multiclass NMS
            nms_boxes, nms_scores, nms_labels = multiclass_nms(
                boxes, pos_scores, pos_labels,
                iou_threshold=args.nms_thresh,
                score_threshold=args.conf_thresh
            )
            
            # Format boxes for output
            image_boxes = []
            for box, score, label in zip(nms_boxes, nms_scores, nms_labels):
                image_boxes.append({
                    "class": classes[label.item()],
                    "confidence": round(float(score.item()), 4),
                    "bbox": [round(float(coord), 2) for coord in box.tolist()]
                })
                
            predictions_json.append({
                "image_id": image_id,
                "boxes": image_boxes
            })
            
    # Export to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions_json, f, ensure_ascii=False, indent=2)
        
    print(f"Predictions successfully written to {output_path}")

if __name__ == "__main__":
    main()
