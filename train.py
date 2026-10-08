import argparse
import json
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add public/tools to sys.path to import evaluate_predictions dynamically
workspace_dir = Path(__file__).resolve().parent
possible_tools_dirs = [
    workspace_dir / "public" / "tools",
    workspace_dir / "final_public" / "public" / "tools",
    Path("public/tools"),
    Path("final_public/public/tools")
]

evaluate_predictions = None
for tools_dir in possible_tools_dirs:
    if tools_dir.exists():
        sys.path.append(str(tools_dir.resolve()))
        try:
            import evaluate_predictions
            if evaluate_predictions is not None:
                print(f"Successfully imported evaluate_predictions from: {tools_dir}")
                break
        except ImportError:
            pass

from utils.dataset import ObjectDetectionDataset, collate_fn
from utils.od_utils import get_fpn_coords, assign_targets, multiclass_nms
from models.detector import FCOS
from models.loss import FCOSLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train Custom FCOS Object Detector")
    parser.add_argument("--train_data", type=str, default="./final_public/public/annotations/train.json", help="Path to train JSON")
    parser.add_argument("--val_data", type=str, default="./final_public/public/annotations/val.json", help="Path to val JSON")
    parser.add_argument("--image_dir", type=str, default="./final_public/public/train/images", help="Path to train images")
    parser.add_argument("--val_image_dir", type=str, default="./final_public/public/val/images", help="Path to val images")
    parser.add_argument("--checkpoint_dir", type=str, default="./models/", help="Directory to save checkpoints")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (reduce to 4 if VRAM is low)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--target_size", type=int, default=512, help="Input resolution")
    parser.add_argument("--multi_scale", action="store_true", help="Enable multi-scale training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--eval_interval", type=int, default=5, help="Interval (in epochs) to evaluate on validation set")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (number of evaluations without improvement)")
    
    return parser.parse_args()

@torch.no_grad()
def evaluate_model(model, val_loader, val_json_path, classes, device, conf_thresh=0.05, nms_thresh=0.5):
    """
    Run inference on validation set and calculate mAP@0.5 using the official evaluation script.
    """
    model.eval()
    predictions_json = []
    
    print("Evaluating validation set...")
    for batch in tqdm(val_loader):
        images = batch["images"].to(device)
        image_ids = batch["image_ids"]
        orig_sizes = batch["orig_sizes"] # [B, 2] (h, w)
        target_sizes = batch["target_sizes"] # [B, 2] (h, w)
        
        outputs = model(images)
        cls_logits = outputs["cls_logits"] # [B, M, num_classes]
        reg_preds = outputs["reg_preds"] # [B, M, 4] (l, t, r, b)
        centerness = outputs["centerness"] # [B, M, 1]
        
        batch_size = images.shape[0]
        h_target, w_target = images.shape[2], images.shape[3]
        
        # Grid coords
        coords = get_fpn_coords(height=h_target, width=w_target, strides=[8, 16, 32], device=device)
        all_x = torch.cat([c["x"] for c in coords]) # [M]
        all_y = torch.cat([c["y"] for c in coords]) # [M]
        all_strides = torch.cat([torch.full_like(c["x"], c["stride"]) for c in coords]) # [M]
        
        for b_idx in range(batch_size):
            img_id = image_ids[b_idx]
            orig_h, orig_w = orig_sizes[b_idx]
            tar_h, tar_w = target_sizes[b_idx]
            
            # 1. Fetch scores, offsets, strides
            b_cls_logits = cls_logits[b_idx] # [M, 5]
            b_reg_preds = reg_preds[b_idx] # [M, 4]
            b_centerness = centerness[b_idx].sigmoid() # [M, 1]
            
            # Classification probability
            b_cls_probs = torch.sigmoid(b_cls_logits) # [M, 5]
            # Final detection scores: class_probs * centerness
            b_scores = b_cls_probs * b_centerness # [M, 5]
            
            # 2. Get class and score for each location
            max_scores, labels = torch.max(b_scores, dim=1) # [M], [M]
            
            # Filter location with score > conf_thresh
            keep_mask = max_scores >= conf_thresh
            if not keep_mask.any():
                predictions_json.append({
                    "image_id": img_id,
                    "boxes": []
                })
                continue
                
            pos_scores = max_scores[keep_mask]
            pos_labels = labels[keep_mask]
            pos_reg = b_reg_preds[keep_mask] # [num_pos, 4]
            
            pos_x = all_x[keep_mask]
            pos_y = all_y[keep_mask]
            pos_strides = all_strides[keep_mask]
            
            # Convert reg predictions back to absolute boxes in target scale
            # l, t, r, b in stride units
            l = pos_reg[:, 0] * pos_strides
            t = pos_reg[:, 1] * pos_strides
            r = pos_reg[:, 2] * pos_strides
            b = pos_reg[:, 3] * pos_strides
            
            xmin = pos_x - l
            ymin = pos_y - t
            xmax = pos_x + r
            ymax = pos_y + b
            
            boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1) # [num_pos, 4]
            
            # Scale coordinates back to original image scale
            scale_x = orig_w / tar_w
            scale_y = orig_h / tar_h
            
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            # Clip boxes
            boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, orig_h)
            
            # 3. Multiclass NMS
            nms_boxes, nms_scores, nms_labels = multiclass_nms(
                boxes, pos_scores, pos_labels, 
                iou_threshold=nms_thresh, 
                score_threshold=conf_thresh
            )
            
            # Format predictions
            image_preds = []
            for box, score, label in zip(nms_boxes, nms_scores, nms_labels):
                image_preds.append({
                    "class": classes[label.item()],
                    "confidence": float(score.item()),
                    "bbox": [float(x) for x in box.tolist()]
                })
                
            predictions_json.append({
                "image_id": img_id,
                "boxes": image_preds
            })
            
    # Calculate mAP
    if evaluate_predictions is not None:
        try:
            with open(val_json_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
                
            classes_eval, image_info = evaluate_predictions.validate_ground_truth(ground_truth)
            normalized_preds = evaluate_predictions.normalize_predictions(
                predictions_json,
                classes=classes_eval,
                image_info=image_info,
                max_detections_per_image=100,
                require_complete=True
            )
            
            results = evaluate_predictions.evaluate(
                ground_truth=ground_truth,
                predictions=normalized_preds,
                classes=classes_eval,
                iou_threshold=0.5
            )
            print(f"Validation mAP@0.5: {results['mAP@0.5']:.4f} (Score points: {results['performance_points']}/20)")
            return results["mAP@0.5"]
        except Exception as e:
            print(f"Failed to calculate official mAP metric: {e}")
            return 0.0
    else:
        print("evaluate_predictions.py not found, skipping mAP calculation.")
        return 0.0

def main():
    args = parse_args()
    
    # Paths validation and setup
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Datasets and Loaders
    print("Loading datasets...")
    train_dataset = ObjectDetectionDataset(
        json_path=args.train_data,
        image_dir=args.image_dir,
        target_size=args.target_size,
        is_train=True,
        multi_scale=args.multi_scale
    )
    val_dataset = ObjectDetectionDataset(
        json_path=args.val_data,
        image_dir=args.val_image_dir,
        target_size=args.target_size,
        is_train=False,
        multi_scale=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    classes = train_dataset.classes
    print(f"Classes: {classes}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # 2. Model, Loss, Optimizer
    model = FCOS(num_classes=len(classes), pretrained=True)
    model.to(device)
    
    criterion = FCOSLoss(alpha=0.25, gamma=2.0, beta=1.0)
    
    # Fine-tuning: optimize backbone with slightly lower learning rate is optional, 
    # but we'll optimize all parameters with lr.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # LR Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 3. Training Loop
    best_map = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        cls_total = 0.0
        reg_total = 0.0
        ctr_total = 0.0
        
        print(f"\n--- Epoch {epoch}/{args.epochs} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")
        
        pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            images = batch["images"].to(device)
            boxes = [b.to(device) for b in batch["boxes"]]
            labels = [l.to(device) for l in batch["labels"]]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Generate FPN coordinates based on current batch image shape
            h, w = images.shape[2], images.shape[3]
            coords = get_fpn_coords(height=h, width=w, strides=[8, 16, 32], device=device)
            
            # Assign targets
            cls_targets, reg_targets, centerness_targets, is_pos = assign_targets(
                coords, boxes, labels, num_classes=len(classes), center_radius=1.5, device=device
            )
            
            targets_dict = {
                "cls_targets": cls_targets,
                "reg_targets": reg_targets,
                "centerness_targets": centerness_targets,
                "is_pos": is_pos
            }
            
            # Calculate loss
            loss, loss_components = criterion(outputs, targets_dict)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            cls_total += loss_components["loss_cls"]
            reg_total += loss_components["loss_reg"]
            ctr_total += loss_components["loss_ctr"]
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_components['loss_cls']:.4f}",
                "reg": f"{loss_components['loss_reg']:.4f}"
            })
            
        # Log training statistics
        num_steps = len(train_loader)
        print(f"Training summary - Epoch {epoch}: Loss: {epoch_loss/num_steps:.4f} (Cls: {cls_total/num_steps:.4f}, Reg: {reg_total/num_steps:.4f}, Ctr: {ctr_total/num_steps:.4f})")
        
        # Step learning rate
        scheduler.step()
        
        # Save checkpoints
        checkpoint_path = checkpoint_dir / "latest.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_map": best_map,
            "classes": classes
        }, checkpoint_path)
        
        # Evaluate model on validation set every eval_interval epochs or on the last epoch
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_map = evaluate_model(model, val_loader, args.val_data, classes, device)
            
            if val_map > best_map:
                best_map = val_map
                best_checkpoint_path = checkpoint_dir / "best.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_map": best_map,
                    "classes": classes
                }, best_checkpoint_path)
                print(f"--> Saved new BEST checkpoint to {best_checkpoint_path} with mAP: {best_map:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{args.paws_check_placeholder if False else args.patience}")
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
            
    print(f"\nTraining completed! Best Validation mAP@0.5: {best_map:.4f}")

if __name__ == "__main__":
    main()
