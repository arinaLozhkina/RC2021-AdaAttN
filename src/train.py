from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

# import source files
from simple_dataset import ContentStyleDataset
from loss import LossCalculate
from models import OverallModel, EncoderModel


class TrainModel(object):
    """
    Overall model training
    """

    def __init__(self):
        # hyper parameters
        self.num_epochs = 10  # number of epochs to train
        self.train_dataset_length = 500  # number of images in train dataset
        self.val_dataset_length = 100  # number of images in validation dataset
        self.checkpoints = 5  # times of checkpoints
        self.batch_size = 8
        self.train_ratio = 0.8  # part of train data in a full dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu or cpu
        self.alpha, self.beta1, self.beta2 = 1e-4, 0.9, 0.999  # Adam parameters
        self.lambda_g, self.lambda_l = 10, 3  # loss parameters
        self.adattn_shape = 64  # squared image size
        self.pretrained = True  # if we want to use pretrained weights
        self.version = 0  # version of checkpoint file with weights

        # Initialisation
        self.Encoder = EncoderModel(self.device)
        self.Model = OverallModel(self.adattn_shape, self.device).to(self.device)  # neural network
        self.Criterion = LossCalculate(self.lambda_l, self.lambda_g, self.device)  # loss function
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr=self.alpha, betas=(self.beta1, self.beta2))
        torch.autograd.set_detect_anomaly(True)

        # Tracking losses
        self.train_loss = []  # [total loss, global loss, local loss] for train
        self.val_loss = []  # [total loss, global loss, local loss] for validation

    def load_data(self):
        """
        Load initial data to dataloader
        :return:
        """
        dataset = ContentStyleDataset(length=self.train_dataset_length + self.val_dataset_length,
                                      shape=self.adattn_shape)
        train_sampler = SubsetRandomSampler(np.arange(self.train_dataset_length))
        val_sampler = SubsetRandomSampler(np.linspace(self.train_dataset_length, self.train_dataset_length +
                                                      self.val_dataset_length - 1, self.val_dataset_length).astype(int))
        self.TrainLoader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)  # create train loader
        self.ValLoader = DataLoader(dataset, batch_size=self.batch_size, sampler=val_sampler)  # create val loader

    def train_epoch(self, epoch=0):
        """
        Model training for current epoch with training dataset
        :return:
        """
        self.Model.train()
        for idx, (content_im, style_im) in enumerate(self.TrainLoader):  # run 1 batch
            content_im, style_im = content_im.to(self.device), style_im.to(self.device)  # convert to GPU
            with torch.set_grad_enabled(True):
                self.Optimizer.zero_grad()
                features = self.Encoder.forward(content_im, style_im)  # get encoding
                I_cs = self.Model(features)  # get predictions
                all_loss, global_loss, local_loss = self.Criterion.total_loss(I_cs, features)  # get loss
                self.train_loss.append(all_loss.item())
                print("Train loss:", all_loss, global_loss, local_loss)
                del global_loss, local_loss
                all_loss.backward()  # backward loss for model fitting
                self.train_loss.append(all_loss.item())  # accumulate losses
                self.Optimizer.step()  # update optimizer

    def validation_epoch(self, epoch=0):
        """
        Test model on a current epoch with validation data
        :params epoch: (Optional) current epoch
        :return:
        """
        self.Model.eval()
        for idx, (content_im, style_im) in enumerate(self.ValLoader):  # run 1 batch
            content_im, style_im = content_im.to(self.device), style_im.to(self.device)  # convert to GPU
            with torch.set_grad_enabled(False):
                features = self.Encoder.forward(content_im, style_im)  # get encoding
                I_cs = self.Model(features)  # get predictions
                all_loss, global_loss, local_loss = self.Criterion.total_loss(I_cs, features)  # get loss
                self.val_loss.append(all_loss.item())
                print("Validation loss:", all_loss, global_loss, local_loss)
                del all_loss, global_loss, local_loss
                if idx == 0:  # on the beginning of the epoch plot results
                    self.plot_result_images(content_im, style_im, I_cs, epoch)


    @staticmethod
    def plot_result_images(content, style, im_cs, epoch=0, directory="./results"):
        """
        Plot content and style initial images and stylized result images and save it to folder results
        :params content: content image
        :params style: content image
        :params im_cs: content image
        :params epoch: (Optional) current epoch
        :params directory: (Optional) path to folder where images are saved
        :return:
        """
        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(content.detach().cpu().numpy()[0, :3, :, :].transpose(1, 2, 0))
        ax[0].axis('off')
        ax[1].imshow(style.detach().cpu().numpy()[0, :3, :, :].transpose(1, 2, 0))
        ax[1].axis('off')
        ax[2].imshow((im_cs.detach().cpu().numpy()[0, :3, :, :].transpose(1, 2, 0) * 255).astype(np.uint8))
        ax[2].axis('off')
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, f"results_{epoch}.png"))

    def train_full(self):
        """
        Run training for all epochs and save weights
        :return:
        """
        self.load_data()  # load data to DataLoaders
        if self.pretrained and os.path.isfile(f"./checkpoint{self.version}.pth"):   # load weights from pretrained model
            checkpoint = torch.load(f"./checkpoint{self.version}.pth", map_location=self.device)
            self.Model.load_state_dict(checkpoint['model_state_dict'])
            self.Optimizer.load_state_dict(checkpoint['optimizer'])

        print("*" * 60 + "Start training" + "*" * 60)
        for epoch in range(self.num_epochs):  # run epoch
            # for every epoch: train -> validation
            self.train_epoch(epoch)
            self.validation_epoch(epoch)
            print(f"Epoch {epoch} / {self.num_epochs}: Train loss {self.train_loss[-1]}, "
                  f"Validation loss: {self.val_loss[-1]}")

            # save results
            print("*" * 60 + "Saving results" + "*" * 60)
            if epoch % (self.num_epochs // self.checkpoints + 1) == 0:
                torch.save({
                    'model_state_dict': self.Model.state_dict(),
                    'optimizer': self.Optimizer.state_dict()
                }, f'checkpoint{epoch // (self.num_epochs // self.checkpoints + 1)}.pth')

        # save losses
        with open('train_loss', 'wb') as fp:
            pickle.dump(self.train_loss, fp)
        with open('val_loss', 'wb') as fp:
            pickle.dump(self.val_loss, fp)


if __name__ == '__main__':
    model = TrainModel()
    model.train_full()
