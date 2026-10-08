import torch
import torch.nn.functional as F

def get_fpn_coords(height, width, strides=[8, 16, 32], device="cpu"):
    """
    Generate grid coordinates for each FPN level.
    Returns:
        coords: list of dicts containing 'x', 'y', 'stride', 'level_idx'
    """
    coords = []
    for level_idx, stride in enumerate(strides):
        h_i = height // stride
        w_i = width // stride
        
        # Grid coordinates in the downsampled space
        y_grid = torch.arange(0, h_i, dtype=torch.float32, device=device)
        x_grid = torch.arange(0, w_i, dtype=torch.float32, device=device)
        y, x = torch.meshgrid(y_grid, x_grid, indexing="ij")
        
        # Project back to input image coordinate system (center of the cell)
        x_img = (x + 0.5) * stride
        y_img = (y + 0.5) * stride
        
        # Flatten
        x_img = x_img.reshape(-1)
        y_img = y_img.reshape(-1)
        
        coords.append({
            "x": x_img,
            "y": y_img,
            "stride": stride,
            "level_idx": level_idx,
            "num_points": x_img.shape[0]
        })
    return coords

def assign_targets(coords, gt_boxes_list, gt_labels_list, num_classes=5, center_radius=1.5, device="cpu"):
    """
    FCOS vectorized target assignment with center sampling.
    Args:
        coords: list of dicts from get_fpn_coords
        gt_boxes_list: list of tensors, each shape [N_gt, 4] (xmin, ymin, xmax, ymax)
        gt_labels_list: list of tensors, each shape [N_gt]
    Returns:
        cls_targets: tensor of shape [B, Total_Locations, Num_Classes]
        reg_targets: tensor of shape [B, Total_Locations, 4] (l, t, r, b) in stride units
        centerness_targets: tensor of shape [B, Total_Locations, 1]
        is_pos: boolean tensor of shape [B, Total_Locations]
    """
    batch_size = len(gt_boxes_list)
    
    # Concatenate all coords from different FPN levels
    all_x = torch.cat([c["x"] for c in coords]) # [M]
    all_y = torch.cat([c["y"] for c in coords]) # [M]
    all_strides = torch.cat([torch.full_like(c["x"], c["stride"]) for c in coords]) # [M]
    all_level_idx = torch.cat([torch.full_like(c["x"], c["level_idx"]) for c in coords]).long() # [M]
    
    num_locs = all_x.shape[0]
    
    # Define size range limits for each level P3, P4, P5
    # P3: [0, 64], P4: [64, 128], P5: [128, 99999]
    limits = [
        [0, 64],
        [64, 128],
        [128, 999999]
    ]
    limits_tensor = torch.tensor(limits, dtype=torch.float32, device=device) # [3, 2]
    
    # Prepare output tensors
    cls_targets = torch.zeros((batch_size, num_locs, num_classes), dtype=torch.float32, device=device)
    reg_targets = torch.zeros((batch_size, num_locs, 4), dtype=torch.float32, device=device)
    centerness_targets = torch.zeros((batch_size, num_locs, 1), dtype=torch.float32, device=device)
    is_pos = torch.zeros((batch_size, num_locs), dtype=torch.bool, device=device)
    
    # Expand coordinates for vectorized computation
    x_loc = all_x.unsqueeze(1) # [M, 1]
    y_loc = all_y.unsqueeze(1) # [M, 1]
    strides = all_strides.unsqueeze(1) # [M, 1]
    level_idx = all_level_idx.unsqueeze(1) # [M, 1]
    
    # Fetch size limits for each location
    loc_min_limit = limits_tensor[all_level_idx, 0].unsqueeze(1) # [M, 1]
    loc_max_limit = limits_tensor[all_level_idx, 1].unsqueeze(1) # [M, 1]
    
    for b_idx in range(batch_size):
        gt_boxes = gt_boxes_list[b_idx] # [N_gt, 4]
        gt_labels = gt_labels_list[b_idx] # [N_gt]
        
        num_gt = gt_boxes.shape[0]
        if num_gt == 0:
            continue
            
        # Reshape gt bboxes
        # gt_boxes: [1, N_gt, 4]
        gt_boxes_exp = gt_boxes.unsqueeze(0)
        xmin = gt_boxes_exp[:, :, 0] # [1, N_gt]
        ymin = gt_boxes_exp[:, :, 1]
        xmax = gt_boxes_exp[:, :, 2]
        ymax = gt_boxes_exp[:, :, 3]
        
        # Calculate distances from locations to 4 boundaries of gt boxes
        # l, t, r, b: [M, N_gt]
        l = x_loc - xmin
        t = y_loc - ymin
        r = xmax - x_loc
        b = ymax - y_loc
        
        # Check if the location is inside the ground truth box
        is_inside = (l > 0) & (t > 0) & (r > 0) & (b > 0) # [M, N_gt]
        
        # Center sampling check: location must be within the center crop of the gt boxes
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        
        radius = center_radius * strides # [M, 1]
        
        c_xmin = torch.max(xmin, cx - radius)
        c_ymin = torch.max(ymin, cy - radius)
        c_xmax = torch.min(xmax, cx + radius)
        c_ymax = torch.min(ymax, cy + radius)
        
        is_in_center = (x_loc >= c_xmin) & (x_loc <= c_xmax) & (y_loc >= c_ymin) & (y_loc <= c_ymax) # [M, N_gt]
        
        # Scale constraints check
        max_reg = torch.max(torch.stack([l, t, r, b], dim=-1), dim=-1)[0] # [M, N_gt]
        is_in_scale = (max_reg >= loc_min_limit) & (max_reg < loc_max_limit) # [M, N_gt]
        
        # Combine conditions
        is_candidate = is_inside & is_in_center & is_in_scale # [M, N_gt]
        
        # If a location matches multiple bboxes, assign it to the one with the smallest area
        areas = (xmax - xmin) * (ymax - ymin) # [1, N_gt]
        areas_expanded = areas.expand(num_locs, -1).clone() # [M, N_gt]
        areas_expanded[~is_candidate] = float("inf")
        
        min_area, min_idx = torch.min(areas_expanded, dim=1) # [M], [M]
        
        # Locations that match at least one gt box
        matched_mask = min_area < float("inf") # [M]
        if not matched_mask.any():
            continue
            
        # Get matching gt box index for each matched location
        pos_loc_idx = torch.where(matched_mask)[0]
        pos_gt_idx = min_idx[matched_mask]
        
        # Update positive indicator
        is_pos[b_idx, pos_loc_idx] = True
        
        # Classification Targets
        # gt_labels[pos_gt_idx] gives the class label for each matched location
        pos_labels = gt_labels[pos_gt_idx]
        cls_targets[b_idx, pos_loc_idx, pos_labels] = 1.0
        
        # Bbox regression targets (l, t, r, b in stride units)
        # Select matching distances
        l_pos = l[pos_loc_idx, pos_gt_idx]
        t_pos = t[pos_loc_idx, pos_gt_idx]
        r_pos = r[pos_loc_idx, pos_gt_idx]
        b_pos = b[pos_loc_idx, pos_gt_idx]
        
        stride_pos = all_strides[pos_loc_idx]
        
        reg_targets[b_idx, pos_loc_idx, 0] = l_pos / stride_pos
        reg_targets[b_idx, pos_loc_idx, 1] = t_pos / stride_pos
        reg_targets[b_idx, pos_loc_idx, 2] = r_pos / stride_pos
        reg_targets[b_idx, pos_loc_idx, 3] = b_pos / stride_pos
        
        # Centerness targets
        min_h = torch.min(l_pos, r_pos)
        max_h = torch.max(l_pos, r_pos)
        min_v = torch.min(t_pos, b_pos)
        max_v = torch.max(t_pos, b_pos)
        
        centerness = torch.sqrt((min_h / max_h) * (min_v / max_v))
        centerness_targets[b_idx, pos_loc_idx, 0] = centerness
        
    return cls_targets, reg_targets, centerness_targets, is_pos

def nms(boxes, scores, iou_threshold):
    """
    Vectorized Non-Maximum Suppression (NMS) written from scratch.
    Args:
        boxes: Tensor of shape [N, 4] (xmin, ymin, xmax, ymax)
        scores: Tensor of shape [N]
        iou_threshold: float
    Returns:
        keep: 1D Tensor of indices to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break
            
        i = order[0].item()
        keep.append(i)
        
        # Calculate intersection
        xx1 = torch.max(x1[order[1:]], x1[i])
        yy1 = torch.max(y1[order[1:]], y1[i])
        xx2 = torch.min(x2[order[1:]], x2[i])
        yy2 = torch.min(y2[order[1:]], y2[i])
        
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        
        # Calculate IoU
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / torch.clamp(union, min=1e-6)
        
        # Filter indices where IoU is less than threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]
        
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def multiclass_nms(boxes, scores, labels, iou_threshold=0.5, score_threshold=0.05, max_detections=100):
    """
    Applies class-aware NMS to prediction boxes.
    Args:
        boxes: Tensor of shape [N, 4] (xmin, ymin, xmax, ymax)
        scores: Tensor of shape [N]
        labels: Tensor of shape [N] (integers)
    Returns:
        keep_boxes: [K, 4]
        keep_scores: [K]
        keep_labels: [K]
    """
    keep_boxes_list = []
    keep_scores_list = []
    keep_labels_list = []
    
    unique_labels = torch.unique(labels)
    for cls in unique_labels:
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        # Threshold by score
        score_mask = cls_scores >= score_threshold
        cls_boxes = cls_boxes[score_mask]
        cls_scores = cls_scores[score_mask]
        
        if cls_boxes.shape[0] == 0:
            continue
            
        # Run class NMS
        keep_idx = nms(cls_boxes, cls_scores, iou_threshold)
        
        keep_boxes_list.append(cls_boxes[keep_idx])
        keep_scores_list.append(cls_scores[keep_idx])
        keep_labels_list.append(torch.full_like(keep_idx, cls.item()))
        
    if len(keep_boxes_list) == 0:
        return (
            torch.empty((0, 4), dtype=torch.float32, device=boxes.device),
            torch.empty((0,), dtype=torch.float32, device=boxes.device),
            torch.empty((0,), dtype=torch.long, device=boxes.device)
        )
        
    keep_boxes = torch.cat(keep_boxes_list, dim=0)
    keep_scores = torch.cat(keep_scores_list, dim=0)
    keep_labels = torch.cat(keep_labels_list, dim=0)
    
    # Sort and keep top max_detections
    if keep_scores.shape[0] > max_detections:
        _, sort_idx = keep_scores.sort(descending=True)
        sort_idx = sort_idx[:max_detections]
        keep_boxes = keep_boxes[sort_idx]
        keep_scores = keep_scores[sort_idx]
        keep_labels = keep_labels[sort_idx]
        
    return keep_boxes, keep_scores, keep_labels
