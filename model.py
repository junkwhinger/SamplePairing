from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet_fc(nn.Module):
    def __init__(self, base_model, nb_classes):
        super(Resnet_fc, self).__init__()

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])

        tmp = OrderedDict()
        tmp['last_conv'] = nn.Conv2d(2048, nb_classes, 1, 1)
        tmp['gap'] = nn.AdaptiveAvgPool2d(1)

        self.classifier_layer = nn.Sequential(tmp)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.classifier_layer(x)
        return x.squeeze(-1).squeeze(-1)


class Resnet_fc_CAM(Resnet_fc):
    def __init__(self, base_model, nb_classes):
        super(Resnet_fc_CAM, self).__init__(base_model, nb_classes)
        self.features = None

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        x = self.classifier_layer(features)
        self.features = features.cpu()
        return x

    def generate_cam(self, cls):

        s_1, s_2 = self.features.size()[2:]
        sl_1 = s_1 - 7 + 1
        sl_2 = s_2 - 7 + 1

        W_conv = self.classifier_layer.last_conv.weight.detach().cpu()
        W_conv_c = W_conv[cls]

        A_conv_c = torch.zeros(1, s_1, s_2)
        for i in range(sl_1):
            for j in range(sl_2):
                a_conv_c = (self.features[:, :, i:i+7, j:j+7] * W_conv_c).sum(1)
                A_conv_c[:, i:i+7, j:j+7] = torch.max(A_conv_c[:, i:i+7, j:j+7], a_conv_c)

        _min = A_conv_c.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
        _max = A_conv_c.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        A_conv_c_normalized = (A_conv_c - _min) / (_max - _min)

        return A_conv_c_normalized