import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


class AdaAttNStar(nn.Module):
    """
    Supervision signal that should be deterministic
    """

    def __init__(self, v_dim, qk_dim):
        super(AdaAttNStar, self).__init__()
        self.softmax = nn.Softmax()

    @staticmethod
    def norm(x):
        """
        Channel-wise mean-variance normalization
        :param x:
        :return:
        """
        # we assume that feature extractor from VGG19 returns tensors of shape: (batch_size, channels, w, h)
        for channel in range(3):
            cur_x = x[:, channel, :, :]
            x[:, channel, :, :] = (cur_x - torch.mean(cur_x, dim=[0, 1, 2])) / torch.std(cur_x, dim=[0, 1, 2])
        return x

    def forward(self, F_c, F_s, F_c_previous, F_s_previous):
        # attention map generation
        Q = self.norm(F_c_previous)
        K = self.norm(F_s_previous)
        V = F_s
        print(Q.shape)  # check transpose dim
        A = self.softmax(torch.matmul(torch.transpose(Q, 1, 2), K))  # attention map

        # attention-weighted mean and standard variance map
        print(A.shape)  # check transpose dim
        A_T = torch.transpose(A, 1, 2)
        M = torch.matmul(V, A_T)  # mean
        S = torch.sqrt(torch.matmul(V * V, A_T) - M * M)  # standard variance

        # adaptive normalization
        F_cs = S * self.norm(F_c) + M
        return F_cs


class LossCalculate(object):
    def __init__(self, lambda_l, lambda_g, Model):
        """
        Calculate of loss function
        :param lambda_l: weight of local loss
        :param lambda_g: weight of global loss
        :param Model: get features to calculate global loss
        """
        self.lambda_l = lambda_l  # weight of local loss
        self.lambda_g = lambda_g  # weight of global loss

        # features of VGG19 layers
        self.encoder = torchvision.models.vgg19(pretrained=True)  # pretrained VGG19
        self.return_nodes = {
            'features.13': 'relu_3',  # relu-3_1
            'features.22': 'relu_4',  # relu-4_1
            'features.31': 'relu_5'  # relu-5_1
        }
        self.features = create_feature_extractor(self.encoder,
                                                 return_nodes=self.return_nodes)  # get features of layers 3, 4, 5
        self.AdaAttN_star = AdaAttNStar(256, 256)  # supervision signal for local loss

    def local_loss(self, I_cs, features):
        """
        Makes the model generates better stylized output for local areas.
        :param I_cs: stylized content image
        :param features: features and previous features for 3, 4, 5 layers for content and style images
        :return:
        """
        [F_s3, F_s4, F_s5, F_c3, F_c4, F_c5, F_s3_previous, F_s4_previous, F_s5_previous, F_c3_previous, F_c4_previous,
         F_c5_previous] = features
        E_values = list(self.features(I_cs).values())
        ada_values = [self.AdaAttN_star(F_c3, F_s3, F_c3_previous, F_s3_previous),
                      self.AdaAttN_star(F_c4, F_s4, F_c4_previous, F_s4_previous),
                      self.AdaAttN_star(F_c5, F_s5, F_c5_previous, F_s5_previous)]
        loss = [torch.cdist(E_values[i], ada_values[i], p=2) for i in range(3)]
        return sum(loss)

    def global_loss(self, I_cs, F_values):
        """
        Distances of mean μ and standard deviation σ between generated image and style image in VGG feature space.
        :param I_cs: stylized content image
        :param F_values: F_s3, F_s4, F_s5 - features of VGG19 ReLU for style images
        :return:
        """
        E_values = list(self.features(I_cs).values())
        loss = [torch.cdist(torch.mean(E_values[i]), torch.mean(F_values[i]), p=2) + \
                torch.cdist(torch.std(E_values[i]), torch.std(F_values[i]), p=2) for i in range(3)]
        return sum(loss)

    def total_loss(self, I_cs, features):
        """
        Combination of local and global losses
        :param I_cs: stylized content image
        :param features: features and previous features for 3, 4, 5 layers for content and style images
        :return: global loss, local loss, total_loss
        """
        [F_s3, F_s4, F_s5, _] = features
        global_loss = self.global_loss(I_cs, [F_s3, F_s4, F_s5])
        local_loss = self.local_loss(I_cs, features)
        total_loss = self.lambda_g * global_loss + self.lambda_l * local_loss
        return global_loss, local_loss, total_loss
