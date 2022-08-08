# Copyright 2022, ByteDance LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
from data import *
from benchmarks import *
from utils import *

def run_alg(data, name, lr, nn_dim_list=[8, 1], freq=4, num_head=2):
    if name == 'Online Learning':
        algorithm = Oco(lr=lr)
    elif name == 'Fourier Learning':
        algorithm = Fourier(lr=lr, cutoff_freq=freq)
    elif name == 'Neural Network Design A':
        algorithm = Dnn(lr=lr, nn_dim_list=nn_dim_list, input_dim=1)
    elif name == 'Neural Network Design B':
        algorithm = Dnn2(lr=lr, nn_dim_list=nn_dim_list, input_dim=1)
    elif name == 'Pluralistic':
        algorithm = MultiHead(num_head=num_head, lr=lr)
    else:
        raise "Unknown Model Type"
    algorithm.learn(data)
    return algorithm.loss_traj

def Task1():
    true_cutoff = 5
    data = Data(cutoff_freq=true_cutoff, N=1000, N_period=50, T=1, noise=0.01)
    oco_best_traj = None
    oco_best_loss = float('Inf')
    oco_best_lr = None
    for oco_lr in np.arange(0.01, 0.4, 0.05):
        oco_traj = run_alg(data=data, name='Online Learning', lr=oco_lr)
        oco_loss = eval_traj(oco_traj, data)
        if oco_loss < oco_best_loss:
            oco_best_loss = oco_loss
            oco_best_traj = oco_traj
            oco_best_lr = oco_lr
    oco = Oco(lr=oco_best_lr)
    oco.loss_traj = oco_best_traj

    dnn1_best_traj, dnn2_best_traj = None, None
    dnn1_best_loss, dnn2_best_loss = float('Inf'), float('Inf')
    dnn1_best_lr, dnn2_best_lr = None, None
    dnn1_best_hd, dnn2_best_hd = None, None
    for dnn_lr in np.arange(0.01, 0.4, 0.05):
        for hidden_dim in [8, 16, 32, 64]:
            dnn1_traj = run_alg(data=data, name='Neural Network Design A', lr=dnn_lr, nn_dim_list=[hidden_dim, 1])
            dnn2_traj = run_alg(data=data, name='Neural Network Design B', lr=dnn_lr, nn_dim_list=[hidden_dim, 1])
            dnn1_loss = eval_traj(dnn1_traj, data)
            dnn2_loss = eval_traj(dnn2_traj, data)
            if dnn1_loss < dnn1_best_loss:
                dnn1_best_loss = dnn1_loss
                dnn1_best_traj = dnn1_traj
                dnn1_best_lr = dnn_lr
                dnn1_best_hd = hidden_dim
            if dnn2_loss < dnn2_best_loss:
                dnn2_best_loss = dnn2_loss
                dnn2_best_traj = dnn2_traj
                dnn2_best_lr = dnn_lr
                dnn2_best_hd = hidden_dim

    dnn1 = Dnn(input_dim=1, lr=dnn1_best_lr, nn_dim_list=[dnn1_best_hd, 1])
    dnn1.loss_traj = dnn1_best_traj
    dnn2 = Dnn2(input_dim=1,lr=dnn2_best_lr, nn_dim_list=[dnn2_best_hd, 1])
    dnn2.loss_traj = dnn2_best_traj

    mh_best_traj = None
    mh_best_loss = float('Inf')
    mh_best_lr = None
    mh_best_hn = None
    for mh_lr in np.arange(0.01, 0.4, 0.05):
        for head_num in [2, 4, 6, 8, 12, 24]:
            mh_traj = run_alg(data=data, name='Pluralistic', lr=mh_lr, num_head=head_num)
            mh_loss = eval_traj(mh_traj, data)
            if mh_loss < mh_best_loss:
                mh_best_traj = mh_traj
                mh_best_loss = mh_loss
                mh_best_lr = mh_lr
                mh_best_hn = head_num
    mh = MultiHead(num_head=mh_best_hn, lr=mh_best_lr)
    mh.loss_traj = mh_best_traj

    fourier_best_traj = None
    fourier_best_loss = float('Inf')
    fourier_best_lr = None
    fourier_best_freq = None
    for fourier_lr in np.arange(0.01, 0.4, 0.05):
        for freq in [2, 6, 10, 14, 18]:
            fourier_traj = run_alg(data=data, name='Fourier Learning', lr=fourier_lr, freq=freq)
            fourier_loss = eval_traj(fourier_traj, data)
            if fourier_loss < fourier_best_loss:
                fourier_best_traj = fourier_traj
                fourier_best_loss = fourier_loss
                fourier_best_lr = fourier_lr
                fourier_best_freq = freq
    fourier = Fourier(lr=fourier_best_lr, cutoff_freq=fourier_best_freq)
    fourier.loss_traj = fourier_best_traj

    algorithms = [oco, dnn1, dnn2, mh, fourier]
    plot_loss_period(data, algorithms)

if __name__ == '__main__':

    Task1()



