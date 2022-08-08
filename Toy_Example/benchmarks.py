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
from utils import *
import tensorflow as tf

class Algorithm(object):
    def __init__(self, name, lr):
        self.name = name
        self.lr = lr
        self.a_traj = []
        self.loss_traj = []
        print("Building algoirthm {x} with default learning rate {y}".format(x=name, y=lr))

    def learn(self, data, **args):
        print("Learning with Algorithm {x}".format(x=self.name))

class Fourier(Algorithm):
    def __init__(self, lr=0.01, cutoff_freq=2):
        super().__init__(name='Fourier Learning', lr=lr)
        self.cutoff_freq = cutoff_freq
        self.T = None
        self.re = [0] * int(cutoff_freq)
        self.im = [0] * int(cutoff_freq)

    def learn(self, data, decay=0.0):
        self.T = data.T
        t = data.data[0]
        x = data.data[1]
        y = data.data[2]
        for i in range(len(t)):
            this_t = t[i]
            this_x = x[i]
            this_y = y[i]
            this_re = [np.cos(2 * np.pi * n * this_t / self.T) for n in range(self.cutoff_freq)]
            this_im = [- np.sin(2 * np.pi * n * this_t / self.T) for n in range(self.cutoff_freq)]
            this_a = np.sum([coef * re for coef, re in zip(self.re, this_re)]) + np.sum([coef * im for coef, im in zip(self.im, this_im)])
            self.a_traj.append(this_a)
            self.loss_traj.append(dist(this_a, data.opt(this_t)))
            grad_re = [2 * (this_a * this_x - this_y) * this_x * re for re in this_re]
            grad_im = [2 * (this_a * this_x - this_y) * this_x * im for im in this_im]
            self.re = [re - self.lr / (1 + np.sqrt(decay * i)) * grad for re, grad in zip(self.re, grad_re)]
            self.im = [im - self.lr / (1 + np.sqrt(decay * i)) * grad for im, grad in zip(self.im, grad_im)]

class Oco(Algorithm):
    def __init__(self, lr=0.01):
        super().__init__(name='Online Learning', lr=lr)
        self.a = 0.0

    def learn(self, data):
        t = data.data[0]
        x = data.data[1]
        y = data.data[2]
        for i in range(len(t)):
            this_t = t[i]
            this_x = x[i]
            this_y = y[i]

            self.a_traj.append(self.a)
            self.loss_traj.append(dist(self.a, data.opt(this_t)))

            grad = 2 * (self.a * this_x - this_y) * this_x
            self.a -= self.lr * grad

class Dnn(Algorithm):
    def __init__(self, input_dim, nn_dim_list, lr=0.01):
        super().__init__(name='Neural Network Design A', lr=lr)
        self.batch_size = 1
        self.x_dim = input_dim
        self.nn_dim_list = nn_dim_list
        self.t_dim = 1
        self.build()

    def build(self):
        self.x = tf.placeholder(tf.float32, shape=(self.batch_size, self.x_dim))
        self.t = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        self.label = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        input = self.t
        with tf.name_scope("MLP"):
            for i, hidden_dim in enumerate(self.nn_dim_list):
                input = tf.layers.dense(
                    inputs=input,
                    units=hidden_dim,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    activation=tf.nn.relu if i != len(self.nn_dim_list)-1 else None)
        self.a = input
        self.output = tf.reduce_sum(input * self.x, axis=-1)
        self.loss = tf.reduce_mean((self.output - self.label) ** 2, axis=0)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.opt_op = self.optimizer.minimize(loss=self.loss)
        self.init_op = tf.global_variables_initializer()

    def learn(self, data):
        t = data.data[0]
        x = data.data[1]
        y = data.data[2]
        max_iter = len(t)
        sess = tf.Session()
        sess.run(self.init_op)
        for i in range(max_iter):
            this_t = np.mod(t[i], data.T)
            this_x = x[i]
            this_y = y[i]
            feed_dict = {self.x: [[this_x]], self.t: [[this_t]], self.label: [[this_y]]}
            a, loss = sess.run([self.a, self.loss], feed_dict=feed_dict)
            self.loss_traj.append(loss[0])
            self.a_traj.append(a[0])
            sess.run([self.opt_op], feed_dict=feed_dict)



class Dnn2(Algorithm):
    def __init__(self, input_dim, nn_dim_list, lr=0.01):
        super().__init__(name="Neural Network Design B", lr=lr)
        self.batch_size = 1
        self.x_dim = input_dim
        self.nn_dim_list = nn_dim_list
        self.t_dim = 1
        self.a_traj = None
        self.build()

    def build(self):
        self.input = tf.placeholder(tf.float32, shape=(self.batch_size, self.x_dim+1))
        self.label = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        input = self.input
        with tf.name_scope("MLP"):
            for i, hidden_dim in enumerate(self.nn_dim_list):
                input = tf.layers.dense(
                    inputs=input,
                    units=hidden_dim,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    activation=tf.nn.relu if i != len(self.nn_dim_list)-1 else None)
        self.output = tf.reduce_sum(input, axis=-1)
        self.loss = tf.reduce_mean((self.output - self.label) ** 2, axis=0)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.opt_op = self.optimizer.minimize(loss=self.loss)
        self.init_op = tf.global_variables_initializer()

    def learn(self, data):
        t = data.data[0]
        x = data.data[1]
        y = data.data[2]
        max_iter = len(t)
        sess = tf.Session()
        sess.run(self.init_op)
        for i in range(max_iter):
            this_t = np.mod(t[i], data.T)
            this_x = x[i]
            this_y = y[i]
            feed_dict = {self.input: [[this_x, this_t]], self.label: [[this_y]]}
            loss = sess.run([self.loss], feed_dict=feed_dict)
            self.loss_traj.append(loss[0])
            sess.run([self.opt_op], feed_dict=feed_dict)

class MultiHead(Algorithm):
    def __init__(self, num_head=2, lr=0.01):
        super().__init__(name="Multi-Head", lr=lr)
        self.num_head = num_head
        self.a = [0] * num_head
        self.T = None

    def learn(self, data):
        self.T = data.T
        interval = self.T / self.num_head
        t = data.data[0]
        x = data.data[1]
        y = data.data[2]
        for i in range(len(t)):
            this_t = np.mod(t[i], self.T)
            head_index = int(this_t / interval)
            this_x = x[i]
            this_y = y[i]
            self.a_traj.append(self.a[head_index])
            self.loss_traj.append(dist(self.a[head_index], data.opt(this_t)))

            grad = 2 * (self.a[head_index] * this_x - this_y) * this_x
            self.a[head_index] -= self.lr * grad