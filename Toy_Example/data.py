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

class Data(object):
    def __init__(self, cutoff_freq, N, N_period, T, noise=0.01):
        self.T = T  # period length
        self.N_period = N_period # number of periods
        self.N = N # samples per period
        self.cutoff_freq = cutoff_freq
        self.noise = noise
        self.bias = 0.5
        self.data = self.generate()
        self.opt = lambda t: sum([np.sin(2 * np.pi * n * t + self.bias) ** 3 for n in range(self.cutoff_freq)]) / (self.cutoff_freq - 1)


    def generate(self):
        t = []
        x = []
        y = []
        for n in range(self.N_period):
            this_t = np.sort(np.random.uniform(0, self.T, size=self.N) + n)
            this_x = np.random.normal(loc=np.sin(2 * np.pi * this_t), scale=0.1)
            this_y = np.array([np.sum([np.sin(2 * np.pi * i * t + self.bias) ** 3 for i in range(self.cutoff_freq)]) / (self.cutoff_freq - 1) for t in this_t]) * this_x
            this_y += np.random.normal(loc=0.0, scale=self.noise, size=this_y.shape)
            t.extend(list(this_t))
            x.extend(list(this_x))
            y.extend(list(this_y))
        return (t, x, y)

