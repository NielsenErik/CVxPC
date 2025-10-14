import torch
from torch import nn

import torch.nn.functional as F



class FocalDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def dice_loss(self, pred, target, smooth=1.0):
        """Dice loss for binary segmentation"""
        pred = torch.sigmoid(pred)   # convert logits → probabilities
        intersection = (pred * target).sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
        return 1 - dice.mean()
    
    def focal_loss(self,pred, target, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)  # probability of correct class
        focal = alpha * (1-pt)**gamma * bce
        return focal.mean()

    def forward(self, pred, target):
        f_loss = self.focal_loss(pred, target)
        d_loss = self.dice_loss(pred, target)
        return 0.5 * f_loss + 0.5 * d_loss

class BCEDiceLoss(nn.Module):
    '''
    In case of pos_weight, it becomes weighted BCE loss.
    '''
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def dice_loss(self, pred, target, smooth=1.0):
        """Dice loss for binary segmentation"""
        pred = torch.sigmoid(pred)   # convert logits → probabilities
        intersection = (pred * target).sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        d_loss = self.dice_loss(pred, target)
        return 0.5 * bce_loss + 0.5 * d_loss
    

# ---------------------------
# 2. Simple U-Net Model
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()

        self.dconv_down1 = DoubleConv(3, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.ConvTranspose2d(512, 512, 2, stride=2)

        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out 
    
    def info(self):
        title = "U-Net"
        class_type = __class__.__name__
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        text = (f"Model: {title} ({class_type})\n"
                f"Total parameters: {num_params}\n"
                f"Trainable parameters: {trainable_params}\n")
        return text
        

class ConvEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: tuple[int], kernel_size: int=3, stride: int=1, padding=3, dropout: float=0.2):
        super(ConvEncoder, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding 
        self.dropout_rate = dropout
        self.n_layers = len(hidden_channels)
        
        self.set_layers()
    
    def set_layers(self):
        self.input_layer = nn.Conv2d(self.input_channels, self.hidden_channels[0], 
                                 kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.activation_in = nn.ReLU()
        self.pool_in = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)
        for i in range(self.n_layers-1):
            setattr(self, f'layer_{i}', nn.Conv2d(self.hidden_channels[i], self.hidden_channels[i+1], 
                                                  kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            setattr(self, f'activation_{i}', nn.ReLU())
            setattr(self, f'max_pool{i}', nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride))
        self.dropout = nn.Dropout(self.dropout_rate)
        # self.pool_out = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride)
        # self.output_layer = nn.Linear(self.hidden_channels[-2], self.hidden_channels[-1])


    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_in(x)
        x = self.pool_in(x)
        for i in range(self.n_layers-1):
            layer_i = getattr(self, f'layer_{i}')
            activation_i = getattr(self, f'activation_{i}')
            x = activation_i(layer_i(x))
            x = self.dropout(x)
            x = getattr(self, f'max_pool{i}')(x)
        return x
    
    def predict(self, model_input):
        return self.forward(model_input)
    
class ConvDecoder(nn.Module):
    def __init__(self, encoder, n_classes: int = 1, kernel_size: int=3, stride: int=1, padding=1):
        super(ConvDecoder, self).__init__()
        self.encoder = encoder
        self.hidden_channels = encoder.hidden_channels[::-1]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_layers = len(self.hidden_channels) - 1
        
        self.set_layers(n_classes)

    def set_layers(self, n_classes):
        for i in range(self.n_layers):
            setattr(
                self,
                f'layer_{i}',
                nn.ConvTranspose2d(
                    in_channels=self.hidden_channels[i],
                    out_channels=self.hidden_channels[i+1],
                    kernel_size=self.kernel_size,
                    stride=2,  # upsampling
                    padding=self.padding,
                    output_padding=1
                )
            )
            setattr(self, f'activation_{i}', nn.ReLU())
        
        # Final segmentation output
        self.output_layer = nn.ConvTranspose2d(
            in_channels=self.hidden_channels[-1],
            out_channels=n_classes,
            kernel_size=1
        )

    def forward(self, x):
        for i in range(self.n_layers):
            layer_i = getattr(self, f'layer_{i}')
            activation_i = getattr(self, f'activation_{i}')
            x = activation_i(layer_i(x))
        x = self.output_layer(x)
        return x
    
class ConvAutoEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: tuple[int], n_classes: int=1, kernel_size: int=3, stride: int=1, padding=9, dropout: float=0.2):
        super(ConvAutoEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        
        self.set_encoder()
        self.set_decoder()

    def set_encoder(self):
        self.encoder = ConvEncoder(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, self.dropout)

    def set_decoder(self):
        self.decoder = ConvDecoder(self.encoder, n_classes=self.n_classes, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def center_crop(self, output, target_shape):
        _, _, h, w = output.shape
        _, _, th, tw = target_shape
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        return output[:, :, start_h:start_h+th, start_w:start_w+tw]


    def forward(self, x):
        input_size = x.shape
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if self.padding > 0:
            decoded = self.center_crop(decoded, input_size)
        return decoded
    
    def predict(self, model_input):
        return self.forward(model_input)
    
    def info(self):
        title = "ConvAutoEncoder"
        class_type = __class__.__name__
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        text = (f"Model: {title} ({class_type})\n"
                f"Total parameters: {num_params}\n"
                f"Trainable parameters: {trainable_params}\n")
        return text

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UNetPlusPlus(nn.Module):
    """UNet++ (Nested U-Net) with optional deep supervision.
    Paper: https://arxiv.org/abs/1807.10165 (for reference)
    This is an encoder-decoder with dense skip connections between same-resolution nodes.
    """
    def __init__(self, in_channels=3, num_classes=1, deep_supervision: bool=False, filters=(64,128,256,512,1024)):
        super().__init__()

        # Up-convolutions
        self.up01 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.up11 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.up21 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.up31 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)

        self.up02 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.up12 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.up22 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)

        self.up03 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.up13 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)

        self.up04 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)

        # Decoder nested convs (dense skip concatenations)
        self.conv01 = DoubleConv(filters[0] + filters[0], filters[0])
        self.conv11 = DoubleConv(filters[1] + filters[1], filters[1])
        self.conv21 = DoubleConv(filters[2] + filters[2], filters[2])
        self.conv31 = DoubleConv(filters[3] + filters[3], filters[3])

        self.conv02 = DoubleConv(filters[0]*2 + filters[0], filters[0])
        self.conv12 = DoubleConv(filters[1]*2 + filters[1], filters[1])
        self.conv22 = DoubleConv(filters[2]*2 + filters[2], filters[2])

        self.conv03 = DoubleConv(filters[0]*3 + filters[0], filters[0])
        self.conv13 = DoubleConv(filters[1]*3 + filters[1], filters[1])

        self.conv04 = DoubleConv(filters[0]*4 + filters[0], filters[0])

        # Final classifiers
        if deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))
        x40 = self.conv40(self.pool3(x30))

        # Decoder with nested dense skip connections
        x01 = self.conv01(torch.cat([x00, self.up01(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up21(x30)], dim=1))
        x31 = self.conv31(torch.cat([x30, self.up31(x40)], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self.up02(x11)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up12(x21)], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self.up22(x31)], dim=1))

        x03 = self.conv03(torch.cat([x00, x01, x02, self.up03(x12)], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up13(x22)], dim=1))

        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up04(x13)], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x01)
            out2 = self.final2(x02)
            out3 = self.final3(x03)
            out4 = self.final4(x04)
            return [out1, out2, out3, out4]
        else:
            return self.final(x04)
