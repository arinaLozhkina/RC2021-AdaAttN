import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

# import source files
from dataset import ContentStyleDataset
from loss import LossCalculate
from models import OverallModel


class TrainModel(object):
    """
    Overall model training
    """
    def __init__(self):
        # hyper parameters
        self.num_epochs = 5  # number of epochs to train
        self.train_dataset_length = 8000  # number of images in train dataset
        self.val_dataset_length = 2000  # number of images in validation dataset
        self.checkpoints = 5  # times of checkpoints
        self.batch_size = 8
        self.train_ratio = 0.8  # part of train data in a full dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu or cpu
        self.alpha, self.beta1, self.beta2 = 0.0001, 0.9, 0.999  # Adam parameters
        self.lambda_g, self.lambda_l = 10, 3  # loss parameters
        self.adattn_shape = 256  # parameters for adaptive attention network, squared image size

        # Initialisation
        self.Model = OverallModel(self.adattn_shape).to(self.device)  # neural network
        self.Criterion = LossCalculate(self.lambda_l, self.lambda_g)  # loss function
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr=self.alpha, betas=(self.beta1, self.beta2))
        self.Scheduler = torch.optim.lr_scheduler.ConstantLR(self.Optimizer)  # learning rate scheduler

        # Tracking losses
        self.train_loss = []  # [total loss] for train
        self.val_loss = []  # [total loss] for validation

    def load_data(self):
        """
        Load initial data to dataloader
        :return:
        """
        train_dataset = ContentStyleDataset(self.train_ratio, mode="train", length=self.train_dataset_length)  # get train data from dataset
        val_dataset = ContentStyleDataset(self.train_ratio, mode="val", length=self.val_dataset_length)  # get train data from dataset
        self.TrainLoader = DataLoader(train_dataset, batch_size=self.batch_size)  # create loader
        self.ValLoader = DataLoader(val_dataset, batch_size=self.batch_size)

    def train_epoch(self):
        """
        Model training for current epoch with training dataset
        :return:
        """
        self.Model.train()
        for (content_im, style_im) in self.TrainLoader:  # run 1 batch
            content_im, style_im = content_im.to(self.device), style_im.to(self.device)  # convert to GPU
            with torch.set_grad_enabled(True):
                I_cs, features = self.Model(content_im, style_im)  # get predictions
                self.Optimizer.zero_grad()
                all_loss = self.Criterion.total_loss(I_cs, features)  # get loss
                self.train_loss.append(all_loss)  # accumulate losses
                all_loss.backward()  # backward loss for model fitting
                self.Optimizer.step()  # update optimizer

    def validation_epoch(self):
        """
        Test model on a current epoch with validation data
        :return:
        """
        self.Model.eval()
        for (content_im, style_im) in self.ValLoader:  # run 1 batch
            content_im, style_im = content_im.to(self.device), style_im.to(self.device)  # convert to GPU
            with torch.set_grad_enabled(False):
                I_cs, features = self.Model(content_im, style_im)  # get predictions
                all_loss = self.Criterion.total_loss(I_cs, features)  # get loss
                self.val_loss.append(all_loss)  # accumulate losses

    def train_full(self):
        """
        Run training for all epochs and save weights
        :return:
        """
        self.load_data()  # load data to DataLoaders
        for epoch in range(self.num_epochs):  # run epoch
            # for every epoch: train -> validation
            self.train_epoch()
            self.validation_epoch()
            print(f"Epoch {epoch} / {self.num_epochs}: Train loss {self.train_loss[-1]}, "
                  f"Validation loss: {self.val_loss[-1]}")
            self.Scheduler.step(epoch)

            # save results
            if epoch % (self.num_epochs // self.checkpoints) == 0:
                torch.save({
                    'model_state_dict': self.Model.state_dict(),
                    'optimizer_state_dict': self.Optimizer.state_dict()
                }, f'checkpoint{epoch // (self.num_epochs // self.checkpoints)}.pth')


if __name__ == '__main__':
    model = TrainModel()
    model.train_full()
