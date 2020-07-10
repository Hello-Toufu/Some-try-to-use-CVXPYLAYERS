"""
Created on Sat Jan  6 15:13:08 2018
single user autoencoder in paper
@author: musicbeer
"""
############################################################################
# Author            :   ZDF
# Created on        :   02/25/2020 Tue
# last modified     :   02/25/2020 Tue
# Description       :
# 1. try cvxpylayers in AE design (work, waring can't be turned off) maybe the problem of the source code of cvxpylayers
############################################################################
import torch
from torch import nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import logging

NUM_EPOCHS = 32
BATCH_SIZE = 32
USE_CUDA = False
"""
(n,k)=(parm1,parm2),n,k refer to paper
"""
parm1 = 2
parm2 = 2
# one-hot coding feature dim
M = 2 ** parm2
k = np.log2(M)
k = int(k)
# compressed feature dim
n_channel = parm1
R = k / n_channel
CHANNEL_SIZE = M
train_num = 800
test_num = 50000


class RTN(nn.Module):
    def __init__(self, in_channels, compressed_dim):
        super(RTN, self).__init__()

        self.in_channels = in_channels
        self.compressed_dim = compressed_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
        # cvxpy-layer
        self.cvxpy_layer = self.create_cvxpylayer()

    def create_cvxpylayer(self):
        # m = self.problem_params['m']
        # num_constraints = self.problem_params['num_constraints']
        amplitude_constraint = 1
        _y = cp.Variable((self.compressed_dim))
        # E = cp.Parameter((num_constraints, m))
        # f = cp.Parameter((num_constraints, 1))
        # G = cp.Parameter((2*m, m))
        # h = cp.Parameter((2*m, m))
        _x = cp.Parameter((self.compressed_dim))

        obj = cp.Minimize(cp.sum_squares(_x-_y))
        cons = [_y >= 0, _y <= amplitude_constraint]
        # cons = [_y <= amplitude_constraint]
        problem = cp.Problem(obj, cons)
        assert problem.is_dpp()
        assert problem.is_dcp()

        layer = CvxpyLayer(problem, parameters = [_x], variables = [_y])
        logging.info("  ---------- CVXPY-LAYER CONSTRUCTED ----------")

        return layer

    def signal_constrraint(self,x):
        solver_args = {
            'mode': 'lsqr',
            'verbose': False,
            # can't stop the warning "WARN: aa_init returned NULL, no acceleration applied."
            # 'verbose': True,
            'max_iters': 1000,
            'eps': 1e-6,
            'use_indirect': False,
            'gpu': False,
            'n_jobs_forward': -1,
            'n_jobs_backward': -1
        }

        # x_constr, = self.cvxpy_layer(x, solver_args={'max_iters': 50000, 'verbose': False})
        x_constr, = self.cvxpy_layer(x, solver_args=solver_args)
        return(x_constr)

    def decode_signal(self, x):
        return self.decoder(x)

    def encode_signal(self, x):
        return self.encoder(x)

    def AWGN(self, x, ebno):
        """ Adding Noise for testing step.
        """
        # Normalization.
        # x = (self.in_channels ** 0.5) * (x / x.norm(dim=-1)[:, None])
        # bit / channel_use
        communication_rate = R
        # Simulated Gaussian noise.
        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * ebno) ** 0.5))
        x += noise
        return x

    def forward(self, x):
        x = self.encoder(x)
        # x_constr, = self.cvxpy_layer(x, solver_args={'max_iters': 50000, 'verbose': False})[0]
        x = self.signal_constrraint(x)
        # Normalization.
        # x = (self.in_channels ** 0.5) * (x / x.norm(dim=-1)[:, None])
        # x = 1 * (x / x.norm(dim=-1)[:, None])
        # 7dBW to SNR.
        training_signal_noise_ratio = 5.01187
        # bit / channel_use
        communication_rate = R
        # Simulated Gaussian noise.
        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        x = x + noise
        x = self.decoder(x)

        return x


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    import torch.utils.data as Data
    # from pytorchtools import EarlyStopping

    model = RTN(CHANNEL_SIZE, compressed_dim=n_channel)

    train_labels = (torch.rand(train_num) * CHANNEL_SIZE).long()
    train_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=train_labels)

    test_labels = (torch.rand(test_num) * CHANNEL_SIZE).long()
    test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
    # DataBase in Pytorch
    # dataset = Data.TensorDataset(data_tensor=train_data, target_tensor=train_labels)
    dataset = Data.TensorDataset(train_data, train_labels)
    # datasettest = Data.TensorDataset(data_tensor=test_data, target_tensor=test_labels)
    datasettest = Data.TensorDataset(test_data, test_labels)
    train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = Data.DataLoader(dataset=datasettest, batch_size=test_num, shuffle=True, num_workers=2)
    # optmizer & Loss
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # initialize the early_stopping object
    patience = 10
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Training
    for epoch in range(NUM_EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(x)
            b_label = Variable(y)  # batch label

            decoded = model(b_x)

            loss = loss_fn(decoded, b_label)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            if step % 100 == 0:
                print('Epoch: ', epoch, 'step', step, '| train loss: %.4f' % loss.data)
                print("\n")
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

    if 1:
        """
        draw curve
        """
        EbNodB_range = list(frange(-5, 8, 0.5))
        ber = [None] * len(EbNodB_range)
        for n in range(0, len(EbNodB_range)):
            EbNo = 10.0 ** (EbNodB_range[n] / 10.0)

            for step, (x, y) in enumerate(test_loader):
                b_x = Variable(x)  # batch x, shape (batch, 28*28)
                b_y = Variable(x)  # batch y, shape (batch, 28*28)
                b_label = Variable(y)  # batch label
                encoder = model.encode_signal(b_x)
                encoder = model.signal_constrraint(encoder)
                encoder = model.AWGN(encoder, EbNo)
                decoder = model.decode_signal(encoder)
                pred = decoder.data.numpy()
                label = b_label.data.numpy()
                pred_output = np.argmax(pred, axis=1)
                no_errors = (pred_output != label)
                no_errors = no_errors.astype(int).sum()
                ber[n] = no_errors / test_num
                print('SNR:', EbNodB_range[n], 'BER1:', ber[n])

        ## ploting ber curve
        import matplotlib.pyplot as plt

        plt.plot(EbNodB_range, ber, 'bo', label='Autoencoder(4,4)')
        plt.yscale('log')
        plt.xlabel('SNR Range')
        plt.ylabel('Block Error Rate')
        plt.grid()
        plt.legend(loc='upper right', ncol=1)
        plt.show()
    else:
        """
        plot
        """

        import matplotlib.pyplot as plt

        test_labels = torch.linspace(0, CHANNEL_SIZE - 1, steps=CHANNEL_SIZE).long()
        test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
        test_data = Variable(test_data)
        x = model.encode_signal(test_data)
        x = (n_channel ** 0.5) * (x / x.norm(dim=-1)[:, None])
        plot_data = x.data.numpy()
        plt.scatter(plot_data[:, 0], plot_data[:, 1])
        plt.axis((-2.5, 2.5, -2.5, 2.5))
        # plt.grid()
        plt.show()