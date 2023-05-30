import torchvision.models as models

from torch import nn

vgg16_true = models.vgg16(weights='IMAGENET1K_V1')
vgg16_false = models.vgg16()

print(vgg16_true)
vgg16_true.add_module("linear", nn.Linear(in_features=1000, out_features=10))
print(vgg16_true)
vgg16_true.classifier.add_module("linear", nn.Linear(in_features=1000, out_features=1000))
print(vgg16_true)
