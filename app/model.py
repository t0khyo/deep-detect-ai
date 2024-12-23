import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size = 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size = 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_pool = torch.mean(x, dim = 1, keepdim = True)
        max_pool, _ = torch.max(x, dim = 1, keepdim = True)
        pool = torch.cat([avg_pool, max_pool], dim = 1)
        attention = self.sigmoid(self.conv(pool))
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16, spatial_kernel_size = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class SiameseResNet(nn.Module):
    def __init__(self, model_name = "resnet50", weights = None):
        super(SiameseResNet, self).__init__()
        self.baseModel = models.resnet50(weights = weights)
        self.attention1 = CBAM(in_channels = 256)
        self.attention2 = CBAM(in_channels = 1024)
        self.baseModel.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.baseModel.fc = nn.Identity()
    def forward(self, x):
        out = self.baseModel.conv1(x)
        out = self.baseModel.bn1(out)
        out = self.baseModel.relu(out)
        out = self.baseModel.maxpool(out)
        out = self.attention1(self.baseModel.layer1(out))
        out = self.baseModel.layer2(out)
        out = self.attention2(self.baseModel.layer3(out))
        out = self.baseModel.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

class LogisticSiameseRegression(nn.Module):
    def __init__(self, model):
        super(LogisticSiameseRegression, self).__init__()
        self.model = model
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace = True),
            nn.Linear(256, 1),
            nn.LeakyReLU(inplace = True)
        )
        self.sigmoid = nn.Sigmoid()
    def forward_once(self, x):
        out = self.model(x)
        out = F.normalize(out, p = 2, dim = 1)
        return out
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        diff = out1 - out2
        out = self.fc(diff)
        out = self.sigmoid(out)
        return out

def load_model(model_path, device):
    siamese_model = SiameseResNet()
    siamese_model = nn.DataParallel(siamese_model).to(device)
    model_rms = LogisticSiameseRegression(siamese_model).to(device)
    model_rms.load_state_dict(torch.load(model_path, map_location=device))
    model_rms.eval()
    return model_rms
