import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DoubleConv3D(nn.Module):
    """
    Double 3D convolution block with instance normalization and LeakyReLU
    Standard building block for 3D U-Net as specified in paper
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            mid_channels (int): Number of intermediate channels (default: out_channels)
            dropout (float): Dropout probability
        """
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownBlock3D(nn.Module):
    """
    Downsampling block with max pooling followed by double convolution
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, dropout=dropout)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock3D(nn.Module):
    """
    Upsampling block with transpose convolution or upsampling + convolution
    """
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bilinear (bool): Use bilinear upsampling instead of transpose conv
            dropout (float): Dropout probability
        """
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, 
                                   mid_channels=in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 
                                       kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder path
            x2: Feature map from encoder path (skip connection)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch between x1 and x2
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2,
                       diff_z // 2, diff_z - diff_z // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate3D(nn.Module):
    """
    3D Attention Gate for focusing on relevant features
    Improves skip connections in U-Net
    """
    
    def __init__(self, gate_channels, in_channels, inter_channels):
        """
        Args:
            gate_channels (int): Number of channels in gate signal (from decoder)
            in_channels (int): Number of channels in input signal (from encoder)
            inter_channels (int): Number of intermediate channels
        """
        super().__init__()
        
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    def forward(self, gate, x):
        """
        Args:
            gate: Gate signal from decoder
            x: Input signal from encoder
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(x)
        
        # Resize gate signal to match input dimensions
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric vessel segmentation
    Optimized for pulmonary vessel segmentation with dual outputs (artery/vein)
    Paper configuration: LeakyReLU + InstanceNorm + patch size (C, 64, 64, 64)
    """
    
    def __init__(self, in_channels=1, num_classes=2, features=[32, 64, 128, 256, 512],
                 bilinear=True, dropout=0.1, attention=False, deep_supervision=False):
        """
        Args:
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes (2 for artery/vein)
            features (list): Number of features in each encoder level
            bilinear (bool): Use bilinear upsampling instead of transpose conv
            dropout (float): Dropout probability
            attention (bool): Use attention gates
            deep_supervision (bool): Enable deep supervision
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.attention = attention
        self.deep_supervision = deep_supervision
        
        # Encoder path
        self.inc = DoubleConv3D(in_channels, features[0], dropout=dropout)
        self.down1 = DownBlock3D(features[0], features[1], dropout=dropout)
        self.down2 = DownBlock3D(features[1], features[2], dropout=dropout)
        self.down3 = DownBlock3D(features[2], features[3], dropout=dropout)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = DownBlock3D(features[3], features[4] // factor, dropout=dropout)
        
        # Attention gates (optional)
        if attention:
            self.att4 = AttentionGate3D(
                gate_channels=features[4] // factor,  # 256
                in_channels=features[3],              # 256
                inter_channels=features[3] // 2       # 128
            )
            self.att3 = AttentionGate3D(
                gate_channels=features[3] // factor,  # 128
                in_channels=features[2],              # 128
                inter_channels=features[2] // 2       # 64
            )
            self.att2 = AttentionGate3D(
                gate_channels=features[2] // factor,  # 64
                in_channels=features[1],              # 64
                inter_channels=features[1] // 2       # 32
            )
            self.att1 = AttentionGate3D(
                gate_channels=features[1] // factor,  # 32
                in_channels=features[0],              # 32
                inter_channels=features[0] // 2       # 16
            )
        
        # Decoder path
        self.up1 = UpBlock3D(features[4], features[3] // factor, bilinear, dropout)
        self.up2 = UpBlock3D(features[3], features[2] // factor, bilinear, dropout)
        self.up3 = UpBlock3D(features[2], features[1] // factor, bilinear, dropout)
        self.up4 = UpBlock3D(features[1], features[0], bilinear, dropout)
        
        # Output layers
        self.outc = nn.Conv3d(features[0], num_classes, kernel_size=1)
        
        # Deep supervision auxiliary outputs (optional)
        if deep_supervision:
            self.aux_out1 = nn.Conv3d(features[3] // factor, num_classes, kernel_size=1)
            self.aux_out2 = nn.Conv3d(features[2] // factor, num_classes, kernel_size=1)
            self.aux_out3 = nn.Conv3d(features[1] // factor, num_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, D, H, W] - expects patch size (C, 64, 64, 64)
        
        Returns:
            dict: Output predictions and auxiliary outputs (if enabled)
        """
        # Encoder path
        x1 = self.inc(x)        # 32 channels
        x2 = self.down1(x1)     # 64 channels
        x3 = self.down2(x2)     # 128 channels
        x4 = self.down3(x3)     # 256 channels
        x5 = self.down4(x4)     # 256 channels (with bilinear) or 512 (without)
        
        # Decoder path with skip connections
        if self.attention:
            x4_att = self.att4(gate=x5, x=x4)  # gate: 256, x: 256
            x = self.up1(x5, x4_att)           # output: 128
            
            x3_att = self.att3(gate=x, x=x3)   # gate: 128, x: 128
            x = self.up2(x, x3_att)            # output: 64
            
            x2_att = self.att2(gate=x, x=x2)   # gate: 64, x: 64
            x = self.up3(x, x2_att)            # output: 32
            
            x1_att = self.att1(gate=x, x=x1)   # gate: 32, x: 32
            x = self.up4(x, x1_att)            # output: 32
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        
        # Main output
        main_out = self.outc(x)
        
        result = {'main': main_out}
        
        # Deep supervision auxiliary outputs
        if self.deep_supervision and self.training:
            # Get intermediate decoder features
            x_up1 = self.up1(x5, x4)  # After first upsampling
            x_up2 = self.up2(x_up1, x3)  # After second upsampling
            x_up3 = self.up3(x_up2, x2)  # After third upsampling
            
            # Auxiliary outputs at different scales
            aux1 = self.aux_out1(x_up1)
            aux2 = self.aux_out2(x_up2)
            aux3 = self.aux_out3(x_up3)
            
            # Resize auxiliary outputs to match main output size
            target_size = main_out.shape[2:]
            aux1 = F.interpolate(aux1, size=target_size, mode='trilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=target_size, mode='trilinear', align_corners=False)
            aux3 = F.interpolate(aux3, size=target_size, mode='trilinear', align_corners=False)
            
            result['aux1'] = aux1
            result['aux2'] = aux2
            result['aux3'] = aux3
        
        return result
    
    def predict(self, x, apply_sigmoid=True):
        """
        Prediction method for inference
        
        Args:
            x: Input tensor [B, C, D, H, W]
            apply_sigmoid: Whether to apply sigmoid activation
        
        Returns:
            dict: Predictions for artery and vein
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            main_out = output['main']
            
            if apply_sigmoid:
                main_out = torch.sigmoid(main_out)
            
            # Split into artery and vein predictions
            artery_pred = main_out[:, 0:1, ...]  # First channel
            vein_pred = main_out[:, 1:2, ...]    # Second channel
            
            return {
                'artery': artery_pred,
                'vein': vein_pred,
                'combined': main_out
            }


def create_unet3d(model_type='standard', **kwargs):
    """
    Factory function to create different 3D U-Net variants
    
    Args:
        model_type (str): 'standard', 'attention'
        **kwargs: Additional arguments for model initialization
    
    Returns:
        nn.Module: Configured 3D U-Net model
    """
    if model_type == 'standard':
        return UNet3D(**kwargs)
    
    elif model_type == 'attention':
        kwargs['attention'] = True
        return UNet3D(**kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test and usage example
if __name__ == '__main__':
    print("Testing 3D U-Net Models for Vessel Segmentation...")
    
    # Test different model variants with paper specifications
    models = {
        'Standard U-Net': create_unet3d('standard', in_channels=1, num_classes=2),
        'Attention U-Net': create_unet3d('attention', in_channels=1, num_classes=2),
        'Deep Supervision U-Net': create_unet3d('standard', in_channels=1, num_classes=2, deep_supervision=True)
    }
    
    # Paper specified input size: (C, 64, 64, 64)
    input_tensor = torch.randn(1, 1, 64, 64, 64)
    
    for name, model in models.items():
        print(f"\n=== {name} ===")
        
        # Count parameters
        num_params = count_parameters(model)
        print(f"Parameters: {num_params:,}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        main_output = output['main']
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {main_output.shape}")
        print(f"Output range: [{main_output.min():.3f}, {main_output.max():.3f}]")
        
        # Test prediction method
        if hasattr(model, 'predict'):
            pred_result = model.predict(input_tensor)
            print(f"Artery prediction shape: {pred_result['artery'].shape}")
            print(f"Vein prediction shape: {pred_result['vein'].shape}")
    
    # Test deep supervision
    print(f"\n=== Deep Supervision Test ===")
    ds_model = create_unet3d('standard', in_channels=1, num_classes=2, deep_supervision=True)
    ds_model.train()  # Deep supervision only works in training mode
    
    with torch.no_grad():
        ds_output = ds_model(input_tensor)
    
    print(f"Main output: {ds_output['main'].shape}")
    if 'aux1' in ds_output:
        print(f"Auxiliary outputs: {len([k for k in ds_output.keys() if k.startswith('aux')])}")
        for key in ds_output.keys():
            if key.startswith('aux'):
                print(f"  {key}: {ds_output[key].shape}")
    
    print(f"\n3D U-Net vessel segmentation model ready!")
    print(f"Configuration: LeakyReLU + InstanceNorm + patch size (1, 64, 64, 64)")
    print(f"Output: 2 channels (artery/vein probability maps)")
    print(f"Loss function: BCE + Dice (ready for clDice integration)")