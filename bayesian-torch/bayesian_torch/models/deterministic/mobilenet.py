import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
__all__ = [
    'MobileNet', 'mobilenet_v2'
]

def mobilenet_v2(pretrained=False, **kwargs):
    """Constructs a MobileNetV2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained, **kwargs)
    
    return model

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        self.model = mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            
if __name__ == "__main__":
    model = MobileNet(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
    
    