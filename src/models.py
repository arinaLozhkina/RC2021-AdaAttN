import torch
import torch.nn as nn
from torch.nn import functional
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


class AdaAttN(nn.Module):
    """
    Adaptive Attention Normalization
    """

    def __init__(self, v_dim, qk_dim):
        super(AdaAttN, self).__init__()
        self.f = nn.Conv2d(qk_dim, qk_dim, (1, 1))
        self.g = nn.Conv2d(qk_dim, qk_dim, (1, 1))
        self.h = nn.Conv2d(v_dim, v_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def norm(x):
        """
        Channel-wise mean-variance normalization
        :param x: current vector of shape (batch_size, channels, w, h)
        :return: normalized vector of shape (batch_size, channels, w, h)
        """
        # we assume that feature extractor from VGG19 returns tensors of shape: (batch_size, channels, w, h)
        for channel in range(3):
            cur_x = x[:, channel, :, :]
            x[:, channel, :, :] = (cur_x - torch.mean(cur_x, dim=[0, 1, 2])) / torch.std(cur_x, dim=[0, 1, 2])
        return x

    def forward(self, F_c, F_s, F_c_previous, F_s_previous):
        # attention map generation
        Q = self.f(self.norm(F_c_previous)).flatten(-2, -1)  # query, shape (batch, channels 1:x, H*W)
        K = self.g(self.norm(F_s_previous)).flatten(-2, -1)  # key, shape (batch, channels 1:x, H*W)
        V = self.h(F_s).flatten(-2, -1)  # value, shape (batch, channels, H*W)
        A = self.softmax(torch.matmul(torch.transpose(Q, 1, 2), K))  # attention map, shape (batch, H*W, H*W)

        # attention-weighted mean and standard variance map
        A_T = torch.transpose(A, 1, 2)  # shape (batch, H*W, channels 1:x)
        M = torch.matmul(V, A_T)  # mean, shape (batch, channels, H*W)
        S = torch.sqrt(torch.matmul(V * V, A_T) - M * M)  # standard variance, shape (batch, channels, H*W)

        # adaptive normalization
        F_cs = S * self.norm(F_c).flatten(-2, -1) + M  # S - scale, M - shift, shape (batch, channels, H*W)
        return F_cs


class Decoder(nn.Module):
    """
    From vectors to content image transformed to particular style
    """

    def __init__(self):
        super(Decoder, self).__init__()
        # preprocessing: (batch, channels, H*W) -> (batch, channels, H, W)
        self.unflatten_f3 = nn.Unflatten(-1, (64, 64))
        self.unflatten_f4 = nn.Unflatten(-1, (32, 32))
        self.unflatten_f5 = nn.Unflatten(-1, (16, 16))

        # Stage F5
        self.upsample_f5 = nn.Upsample(scale_factor=2)
        self.conv_f5 = nn.Conv2d(512, 512, (1, 1))
        self.relu_f5 = nn.ReLU()

        # Stage F4
        self.conv_f4 = nn.Conv2d(512, 256, (1, 1))
        self.relu_f4 = nn.ReLU()
        self.upsample_f4 = nn.Upsample(scale_factor=2)

        # Stage F3
        self.conv1_f3 = nn.Conv2d(256, 256, (1, 1))
        self.relu1_f3 = nn.ReLU()
        self.conv2_f3 = nn.Conv2d(256, 256, (1, 1))
        self.relu2_f3 = nn.ReLU()
        self.conv3_f3 = nn.Conv2d(256, 256, (1, 1))
        self.relu3_f3 = nn.ReLU()
        self.conv_f3 = nn.Conv2d(256, 128, (1, 1))
        self.relu_f3 = nn.ReLU()
        self.upsample_f3 = nn.Upsample(scale_factor=2)

        # Stage F2
        self.conv1_f2 = nn.Conv2d(128, 128, (1, 1))
        self.relu1_f2 = nn.ReLU()
        self.conv2_f2 = nn.Conv2d(128, 64, (1, 1))
        self.relu2_f2 = nn.ReLU()
        self.upsample_f2 = nn.Upsample(scale_factor=2)

        # Stage F1
        self.conv_f1 = nn.Conv2d(64, 64, (1, 1))
        self.relu_f1 = nn.ReLU()
        self.conv = nn.Conv2d(64, 3, (1, 1))

    def forward(self, F_cs3, F_cs4, F_cs5):
        # preprocessing
        F_cs3, F_cs4, F_cs5 = self.unflatten_f3(F_cs3), self.unflatten_f4(F_cs4), self.unflatten_f5(F_cs5)

        # Stage F5
        x = self.upsample_f5(F_cs5)
        x = x + F_cs4
        x = self.conv_f5(x)
        x = self.relu_f5(x)
        # print("(512, H / 8, W / 8)", x.shape)

        # Stage F4
        x = self.conv_f4(x)
        x = self.relu_f4(x)
        x = self.upsample_f4(x)
        # print("(256, H / 4, W / 4)", x.shape)

        # Stage F3
        x = x + F_cs3  # torch.concat([x, F_cs3], dim=0)
        x = self.conv1_f3(x)
        x = self.relu1_f3(x)
        x = self.conv2_f3(x)
        x = self.relu2_f3(x)
        x = self.conv3_f3(x)
        x = self.relu3_f3(x)
        x = self.conv_f3(x)
        x = self.relu_f3(x)
        x = self.upsample_f3(x)
        # print("(128, H / 2, W / 2)", x.shape)

        # Stage F2
        x = self.conv1_f2(x)
        x = self.relu1_f2(x)
        x = self.conv2_f2(x)
        x = self.relu2_f2(x)
        x = self.upsample_f2(x)
        # print("(64, H, W)", x.shape)

        # Stage F1
        x = self.conv_f1(x)
        x = self.relu_f1(x)
        x = self.conv(x)
        # print("(3, H, W)", x.shape)
        return x


class OverallModel(nn.Module):
    """
    Full model: encoder (pretrained VGG19) -> get ReLU-3_1, ReLU-4_1, ReLU-5_1 -> AdaAttn -> decoder
    """

    def __init__(self, v_dim):
        super(OverallModel, self).__init__()
        self.encoder = torchvision.models.vgg19(pretrained=True)  # pretrained VGG19
        self.layers_index = [13, 22, 31]  # index of relu-3_1, relu-4_1 and relu-5_1 features in VGG19
        self.return_nodes = {f'features.{i}': f'features.{i}' for i in self.layers_index}  # features of relu
        self.features = create_feature_extractor(self.encoder,
                                                 return_nodes=self.return_nodes)  # get features of relu 3, 4, 5
        self.return_nodes_previous = {f'features.{i}': f'features.{i}' for i in range(31) if i not in self.layers_index}
        self.features_previous = create_feature_extractor(self.encoder,
                                                          return_nodes=self.return_nodes_previous)  # get features of previous layers
        self.AdaAttn = [AdaAttN(v * v_dim, q_dim * v * v_dim) for (v, q_dim) in [(1, 13), (2, 22), (2, 30)]]  # initialize Adaptive attention network
        self.decoder = Decoder()  # initialize decoder

    @staticmethod
    def bilinear_interpolation(x, F):
        """
        Bilinear interpolation D which down-samples the input feature to the shape of F_3/4/5
        :param x: tensor, current layer's feature
        :return:
        """
        spatial_interpolation = functional.interpolate(x, F.shape[2:], mode="bilinear", align_corners=True).permute(0,2, 1, 3)
        channel_interpolation = functional.interpolate(spatial_interpolation, F.shape[1:3], mode='bilinear',
                                                       align_corners=True).permute(0, 2, 1, 3)

        return channel_interpolation

    def get_previous(self, image, F_3, F_4, F_5):
        """
        Concatenate features of current layer with down-sampled features of its previous layers by channel
        :param image: current batch
        :return: concatenation along channels of D(F_i), where F - features of layer i, D - bilinear interpolation
        """
        all_previous = list(self.features_previous(image).values())  # get all previous features
        previous_by_idx = lambda k, f_i: torch.cat([*list(map(self.bilinear_interpolation, all_previous[:k],
                                        [f_i] * k)), f_i], dim=1)  # bilinear interpolation concatenated by channel
        F_3_previous = previous_by_idx(12, F_3)  # from 1 to 13 layers (ReLU-3_1 - 13th layer)
        F_4_previous = previous_by_idx(21, F_4)  # from 1 to 22 layers (ReLU-4_1 - 22th layer)
        F_5_previous = previous_by_idx(29, F_5)  # from 1 to 31 layers (ReLU-5_1 - 31th layer)
        return F_3_previous, F_4_previous, F_5_previous

    def forward(self, content, style):
        # get encoder features
        self.F_c3, self.F_c4, self.F_c5 = self.features(content).values()
        self.F_s3, self.F_s4, self.F_s5 = self.features(style).values()
        # F*^{1:x}
        self.F_c3_previous, self.F_c4_previous, self.F_c5_previous = self.get_previous(content, self.F_c3, self.F_c4,
                                                                                       self.F_c5)
        self.F_s3_previous, self.F_s4_previous, self.F_s5_previous = self.get_previous(style, self.F_s3, self.F_s4,
                                                                                       self.F_s5)

        # adaptive attention network
        F_cs3 = self.AdaAttn[0](self.F_c3, self.F_s3, self.F_c3_previous, self.F_s3_previous)  # for ReLU - 3
        F_cs4 = self.AdaAttn[1](self.F_c4, self.F_s4, self.F_c4_previous, self.F_s4_previous)  # for ReLU - 4
        F_cs5 = self.AdaAttn[2](self.F_c5, self.F_s5, self.F_c5_previous, self.F_s5_previous)  # for ReLU - 5

        # decoder
        I_cs = self.decoder(F_cs3, F_cs4, F_cs5)
        return I_cs, [self.F_s3, self.F_s4, self.F_s5, self.F_c3, self.F_c4, self.F_c5,
                      self.F_s3_previous, self.F_s4_previous, self.F_s5_previous,
                      self.F_c3_previous, self.F_c4_previous, self.F_c5_previous]


# if __name__ == '__main__':
    # model = OverallModel(256)
    # content_im = torch.randn((8, 3, 256, 256))
    # style_im = torch.randn((8, 3, 256, 256))
    # print(model(content_im, style_im))
    # model = AdaAttN(256, 256 * 13)
    # vect = [torch.randn([8, 256, 64, 64]),  torch.randn([8, 256, 64, 64]), torch.randn([8, 3328, 64, 64]), torch.randn([8, 3328, 64, 64])]
    # print(model(*vect))
    # model = Decoder()
    # f3 = torch.randn((8, 256, 4096))
    # f4 = torch.randn((8, 512, 1024))
    # f5 = torch.randn((8, 512, 256))
    # print(model(f3, f4, f5).shape)
