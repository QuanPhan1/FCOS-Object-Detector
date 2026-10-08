import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class FCOSBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained ResNet50
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        
        # Split ResNet stages
        self.stage1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        ) # output channels: 256, stride: 4
        
        self.stage2 = resnet.layer2 # output channels: 512, stride: 8 (C3)
        self.stage3 = resnet.layer3 # output channels: 1024, stride: 16 (C4)
        self.stage4 = resnet.layer4 # output channels: 2048, stride: 32 (C5)
        
    def forward(self, x):
        x = self.stage1(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5

class FPN(nn.Module):
    def __init__(self, in_channels_list=[512, 1024, 2048], out_channels=256):
        super().__init__()
        self.lateral5 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, c3, c4, c5):
        p5_lat = self.lateral5(c5)
        p5 = self.smooth5(p5_lat)
        
        p4_lat = self.lateral4(c4) + F.interpolate(p5_lat, scale_factor=2, mode="bilinear", align_corners=False)
        p4 = self.smooth4(p4_lat)
        
        p3_lat = self.lateral3(c3) + F.interpolate(p4_lat, scale_factor=2, mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3_lat)
        
        return p3, p4, p5

class FCOSHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=5, num_convs=4):
        super().__init__()
        
        # Classification branch
        cls_layers = []
        for _ in range(num_convs):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            cls_layers.append(nn.GroupNorm(32, in_channels))
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_convs = nn.Sequential(*cls_layers)
        
        # Regression branch
        reg_layers = []
        for _ in range(num_convs):
            reg_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            reg_layers.append(nn.GroupNorm(32, in_channels))
            reg_layers.append(nn.ReLU(inplace=True))
        self.reg_convs = nn.Sequential(*reg_layers)
        
        # Predictions
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Scale parameters for each scale level (FCOS standard)
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(3)])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Focal Loss classification bias initialization (p = 0.01)
        bias_val = -float(torch.log(torch.tensor((1 - 0.01) / 0.01)))
        nn.init.constant_(self.cls_logits.bias, bias_val)

    def forward(self, features):
        """
        Args:
            features: list of features [P3, P4, P5]
        Returns:
            cls_scores: list of [B, num_classes, H_i, W_i]
            reg_preds: list of [B, 4, H_i, W_i]
            centerness_scores: list of [B, 1, H_i, W_i]
        """
        cls_scores = []
        reg_preds = []
        centerness_scores = []
        
        for idx, x in enumerate(features):
            cls_feat = self.cls_convs(x)
            reg_feat = self.reg_convs(x)
            
            cls_score = self.cls_logits(cls_feat)
            centerness = self.centerness(cls_feat)
            
            # Predict offsets and scale using level-specific learnable scale parameter
            reg_offset = self.reg_pred(reg_feat)
            reg_offset = torch.exp(self.scales[idx] * reg_offset)
            
            cls_scores.append(cls_score)
            reg_preds.append(reg_offset)
            centerness_scores.append(centerness)
            
        return cls_scores, reg_preds, centerness_scores

class FCOS(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = FCOSBackbone(pretrained=pretrained)
        self.fpn = FPN(in_channels_list=[512, 1024, 2048], out_channels=256)
        self.head = FCOSHead(in_channels=256, num_classes=num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input batch image [B, 3, H, W]
        Returns:
            Dict containing:
                "cls_logits": concatenated [B, Total_Locations, num_classes]
                "reg_preds": concatenated [B, Total_Locations, 4]
                "centerness": concatenated [B, Total_Locations, 1]
        """
        c3, c4, c5 = self.backbone(x)
        features = self.fpn(c3, c4, c5) # [P3, P4, P5]
        cls_scores, reg_preds, centerness_scores = self.head(features)
        
        # Flatten and concatenate predictions across all FPN levels
        batch_size = x.shape[0]
        
        flat_cls = []
        flat_reg = []
        flat_center = []
        
        for cls_lvl, reg_lvl, cent_lvl in zip(cls_scores, reg_preds, centerness_scores):
            # cls_lvl: [B, C, H_i, W_i] -> [B, H_i * W_i, C]
            flat_cls.append(cls_lvl.permute(0, 2, 3, 1).reshape(batch_size, -1, cls_lvl.shape[1]))
            # reg_lvl: [B, 4, H_i, W_i] -> [B, H_i * W_i, 4]
            flat_reg.append(reg_lvl.permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
            # cent_lvl: [B, 1, H_i, W_i] -> [B, H_i * W_i, 1]
            flat_center.append(cent_lvl.permute(0, 2, 3, 1).reshape(batch_size, -1, 1))
            
        cls_logits = torch.cat(flat_cls, dim=1)
        reg_preds = torch.cat(flat_reg, dim=1)
        centerness = torch.cat(flat_center, dim=1)
        
        return {
            "cls_logits": cls_logits,
            "reg_preds": reg_preds,
            "centerness": centerness
        }
