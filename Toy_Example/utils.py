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
import matplotlib

def dist(x,y):
    return (x - y) ** 2

def eval_traj(traj, data):
    # evaluate the trajectory value of the last 5 periods
    traj_period = list(np.mean(np.array(traj).reshape(data.N_period, data.N), axis=-1))
    traj_period = traj_period[-5:]
    return sum(traj_period)/len(traj_period)

def plot_a_traj(data, algorithms):
    fig = plt.figure()
    for algorithm in algorithms:
        plt.plot(algorithm.a_traj, alpha=0.5, label=algorithm.name)
    plt.plot([data.opt(t) for t in data.data[0]], alpha=1.0, linestyle='dashed', linewidth=2, label='Ground Truth')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Model Parameter")
    plt.savefig("./figs/toy_a.eps")
    plt.show()

def plot_loss(data, oco, fourier):
    fig = plt.figure()
    plt.plot(np.log(oco.loss_traj), alpha=1, linewidth=2, label='Online Learning')
    for f in fourier:
        plt.plot(np.log(f.loss_traj), alpha=0.5, label="Fourier Learning, N={i}".format(i=f.cutoff_freq))
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("L2 Distance between Learned and True Model Parameter (Log)")
    plt.savefig("./figs/toy_loss.eps")
    plt.show()

def plot_loss_period(data, algorithms):
    fig = plt.figure()
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    matplotlib.rc('font', size=14)
    for algo in algorithms:
        loss = list(np.mean(np.array(algo.loss_traj).reshape(data.N_period, data.N), axis=-1))
        plt.plot(np.log(loss), alpha=1, linewidth=2, label=algo.name)
    plt.legend()
    plt.grid()
    plt.xlabel("Number of Periods")
    plt.ylabel(r"Averaged Estimation Error of $a$")
    plt.savefig("./figs/toy_loss_against_period.eps")
    plt.show()