import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


class AdaAttNStar(nn.Module):
    """
    Supervision signal that should be deterministic
    """

    def __init__(self, eps=1e-12):
        super(AdaAttNStar, self).__init__()
        self.eps = eps  # small additive value in order not to divide by zero
        # self.softmax = nn.Softmax(dim=-1)

    def norm(self, x):
        """
        Channel-wise mean-variance normalization
        :param x:
        :return:
        """
        # we assume that feature extractor from VGG19 returns tensors of shape: (batch_size, channels, w, h)
        for channel in range(3):
            cur_x = x[:, channel, :, :]
            x[:, channel, :, :] = (cur_x - torch.mean(cur_x, dim=[0, 1, 2])) / (torch.std(cur_x, dim=[0, 1, 2]) + self.eps)
        return x

    def forward(self, F_c, F_s, F_c_previous, F_s_previous):
        # attention map generation
        Q = self.norm(F_c_previous).flatten(-2, -1)  # query, shape (batch, channels 1:x, H*W)
        K = self.norm(F_s_previous).flatten(-2, -1)  # key, shape (batch, channels 1:x, H*W)
        V = F_s.flatten(-2, -1)  # value, shape (batch, channels, H*W)
        A = torch.matmul(torch.transpose(Q, 1, 2), K)  # attention map, shape (batch, H*W, H*W)  # self.softmax(
        # attention-weighted mean and standard variance map
        A_T = torch.transpose(A, 1, 2)  # shape (batch, H*W, channels 1:x)
        M = torch.matmul(V, A_T)  # mean, shape (batch, channels, H*W)
        S = torch.sqrt(torch.matmul(V * V, A_T) - M * M)  # standard variance, shape (batch, channels, H*W)
        # adaptive normalization
        F_cs = torch.nan_to_num(S) * self.norm(F_c).flatten(-2, -1) + M  # S - scale, M - shift, shape (batch, channels, H*W)
        return F_cs


class LossCalculate(object):
    def __init__(self, lambda_l, lambda_g):
        super(LossCalculate, self).__init__()
        """
        Calculate of loss function
        :param lambda_l: weight of local loss
        :param lambda_g: weight of global loss
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
        self.AdaAttN_star = AdaAttNStar()  # supervision signal for local loss

    def local_loss(self, features):
        """
        Makes the model generates better stylized output for local areas.
        :param features: features and previous features for 3, 4, 5 layers for content and style images
        :return:
        """
        [F_s3, F_s4, F_s5, F_c3, F_c4, F_c5, F_s3_previous, F_s4_previous, F_s5_previous, F_c3_previous, F_c4_previous,
         F_c5_previous] = features
        E_values = [elem.flatten(-2, -1) for elem in self.E_values]
        ada_values = [self.AdaAttN_star(F_c3, F_s3, F_c3_previous, F_s3_previous),
                      self.AdaAttN_star(F_c4, F_s4, F_c4_previous, F_s4_previous),
                      self.AdaAttN_star(F_c5, F_s5, F_c5_previous, F_s5_previous)]
        loss = [torch.linalg.matrix_norm(E_values[i] - ada_values[i], dim=[1, 2], ord=2) for i in range(3)]
        return sum(loss)

    def global_loss(self, F_values):
        """
        Calculates distances of mean μ and standard deviation σ between generated image and style image in VGG feature space.
        :param F_values: F_s3, F_s4, F_s5 - features of VGG19 ReLU for style images
        :return:
        """
        loss = [torch.linalg.matrix_norm(torch.mean(self.E_values[i], dim=1) - torch.mean(F_values[i], dim=1), ord=2) +
                torch.linalg.matrix_norm(torch.std(self.E_values[i], dim=1) - torch.std(F_values[i], dim=1), ord=2) for
                i in range(3)]
        return sum(loss)

    def total_loss(self, I_cs, features):
        """
        Combinates local and global losses
        :param I_cs: stylized content image
        :param features: features and previous features for 3, 4, 5 layers for content and style images
        :return: global loss, local loss, total_loss
        """
        self.E_values = list(self.features(I_cs).values())  # calculate features from VGG19 for result image
        [F_s3, F_s4, F_s5, F_c3, F_c4, F_c5, F_s3_previous, F_s4_previous, F_s5_previous, F_c3_previous, F_c4_previous,
         F_c5_previous] = features
        global_loss = self.global_loss([F_s3, F_s4, F_s5])
        local_loss = self.local_loss(features)
        total_loss = self.lambda_g * global_loss + self.lambda_l * local_loss
        return global_loss, local_loss, total_loss


if __name__ == '__main__':
    model = LossCalculate(2, 2)
    I_cs = torch.randn([8, 3, 256, 256])
    features = \
        [torch.randn([8, 256, 64, 64]), torch.randn([8, 512, 32, 32]),
         torch.randn([8, 512, 16, 16]), torch.randn([8, 256, 64, 64]),
         torch.randn([8, 512, 32, 32]), torch.randn([8, 512, 16, 16]),
         torch.randn([8, 3328, 64, 64]),
         torch.randn([8, 11264, 32, 32]),
         torch.randn([8, 15360, 16, 16]),
         torch.randn([8, 3328, 64, 64]),
         torch.randn([8, 11264, 32, 32]),
         torch.randn([8, 15360, 16, 16])]
    print(model.total_loss(I_cs, features))

