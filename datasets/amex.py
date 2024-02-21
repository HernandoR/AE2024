'''
Author: HernandoR lzhen.dev@outlook.com
CreateDate: Do not edit
LastEditors: HernandoR lzhen.dev@outlook.com
LastEditTime: 2024-02-21
Description:

Copyright (c) 2024 by HernandoR lzhen.dev@outlook.com, All Rights Reserved.
'''
"""
An example for dataset loaders, starting with data loading including all the functions that either preprocess or postprocess data.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset


class AmexDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        self.logger.info("Loading Amex DATA.....")
        if config.data_mode == "numpy_train":
            raise NotImplementedError("This mode is not implemented YET")

        elif config.data_mode == "numpy_test":
            raise NotImplementedError("This mode is not implemented YET")
            # test_data = torch.from_numpy(np.load(config.data_folder + config.x_test)).float()
            # test_labels = torch.from_numpy(np.load(config.data_folder + config.y_test)).int()

            # self.len_test_data = test_data.size()[0]

            # self.test_iterations = (self.len_test_data + self.config.batch_size - 1) // self.config.batch_size

            # self.logger.info("""
            #     Some Statistics about the testing data
            #     test_data shape: {}, type: {}
            #     test_labels shape: {}, type: {}
            #     test_iterations: {}
            # """.format(test_data.size(), test_data.type(), test_labels.size(), test_labels.type(),
            #            self.test_iterations))

            # test = TensorDataset(test_data, test_labels)

            # self.test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False)


        elif config.data_mode == "random":
            train_data = torch.randn(self.config.batch_size, self.config.input_channels)
            train_labels = torch.ones(self.config.batch_size).int()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, batch, epoch):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=4,
                           padding=2,
                           normalize=True)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                pass

        imageio.mimsave(self.config.out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass

