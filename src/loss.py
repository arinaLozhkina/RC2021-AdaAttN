import torch
from torch.nn import functional
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


class LossCalculate(object):
    def __init__(self, lambda_l, lambda_g, device):
        super(LossCalculate, self).__init__()
        """
        Calculate of loss function
        :param lambda_l: weight of local loss
        :param lambda_g: weight of global loss
        """
        self.lambda_l = lambda_l  # weight of local loss
        self.lambda_g = lambda_g  # weight of global loss

        # features of VGG19 layers
        self.softmax = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=-1) in the paper
        self.encoder = torchvision.models.vgg19(pretrained=True).to(device).eval()  # pretrained VGG19
        self.return_nodes = {
            'features.13': 'relu_3',  # relu-3_1
            'features.22': 'relu_4',  # relu-4_1
            'features.31': 'relu_5'  # relu-5_1
        }
        self.features = create_feature_extractor(self.encoder,
                                                 return_nodes=self.return_nodes)  # get features of layers 3, 4, 5

    def norm(self, x, eps=1e-12):
        """
        Channel-wise mean-variance normalization
        :param x: Tensor of shape: (batch_size, channels, w, h)
        :param eps: additive value in order not to divide by zero
        :return:
        """
        mean, std = torch.mean(x, dim=[0, 2, 3]), torch.std(x, dim=[0, 2, 3]) + eps
        res = torchvision.transforms.Normalize(mean, std)(x)
        return res

    def adatnn_star(self, F_c, F_s, F_c_previous, F_s_previous):
        """
        Adaptive Attention Normalization Star (defined signal)
        """
        Q = self.norm(F_c_previous).flatten(-2, -1)  # query, shape (batch, channels 1:x, H*W)
        K = self.norm(F_s_previous).flatten(-2, -1)  # key, shape (batch, channels 1:x, H*W)
        V = F_s.flatten(-2, -1)  # value, shape (batch, channels, H*W)
        A = self.softmax(torch.matmul(torch.transpose(Q, 1, 2), K))  # attention map, shape (batch, H*W, H*W)  
        A_T = torch.transpose(A, 1, 2)  # shape (batch, H*W, channels 1:x)
        M = torch.matmul(V, A_T)  # mean, shape (batch, channels, H*W)
        var = torch.matmul(V * V, A_T) - M * M
        S = torch.sqrt(var.clamp(min=0) + 1e-8)  # standard variance, shape (batch, channels, H*W)
        F_cs = torch.nan_to_num(S) * self.norm(F_c).flatten(-2, -1) + M  # S - scale, M - shift
        return F_cs

    def local_loss(self, features_values):
        """
        Makes the model generates better stylized output for local areas.
        :param features_values: features and previous features for 3, 4, 5 layers for content and style images
        :return:
        """
        [F_s3, F_s4, F_s5, F_c3, F_c4, F_c5, F_s3_previous, F_s4_previous, F_s5_previous, F_c3_previous, F_c4_previous,
         F_c5_previous] = features_values
        E_values = [elem.flatten(-2, -1) for elem in self.E_values]
        ada_values = [self.adatnn_star(F_c3, F_s3, F_c3_previous, F_s3_previous),
                      self.adatnn_star(F_c4, F_s4, F_c4_previous, F_s4_previous),
                      self.adatnn_star(F_c5, F_s5, F_c5_previous, F_s5_previous)]
        loss = [functional.mse_loss(E_values[i], ada_values[i]) for i in range(3)]
        return sum(loss)

    def global_loss(self, F_values):
        """
        Calculates distances of mean μ and standard deviation σ between generated image and style image in VGG feature space.
        :param F_values: F_s3, F_s4, F_s5 - features of VGG19 ReLU for style images
        :return:
        """
        loss = [torch.linalg.matrix_norm(torch.mean(self.E_values[i], dim=[1, 2]) - torch.mean(F_values[i], dim=[1, 2]), ord=2) +
                torch.linalg.matrix_norm(torch.std(self.E_values[i], dim=[1, 2]) - torch.std(F_values[i], dim=[1, 2]), ord=2) for
                i in range(3)]
        return sum(loss)

    def total_loss(self, I_cs, features_values):
        """
        Combinates local and global losses
        :param I_cs: stylized content image
        :param features_values: features and previous features for 3, 4, 5 layers for content and style images
        :return: global loss, local loss, total_loss
        """
        self.E_values = list(self.features(I_cs).values())  # calculate features from VGG19 for result image
        [F_s3, F_s4, F_s5, F_c3, F_c4, F_c5, F_s3_previous, F_s4_previous, F_s5_previous, F_c3_previous, F_c4_previous,
         F_c5_previous] = features_values
        local_loss = self.local_loss(features_values)
        global_loss = self.global_loss([F_s3, F_s4, F_s5])
        total_loss = self.lambda_l * local_loss + self.lambda_g * global_loss
        return total_loss, global_loss, local_loss

