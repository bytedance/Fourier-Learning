#!/usr/bin/env bash
# Copyright 2019, The TensorFlow Federated Authors.
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

"""
Note: This file may have been modified by ByteDance Inc.
Experiment settings have been changed to adapt to Fourier Learning paper experiments 
"""
set -e

BAZEL=bazel

#### Constants ####
ROOT="/tmp/sc_paper"
ROOTDATA="${ROOT}/data"
ROOTRAWDATA="${ROOT}/raw_data"
LOGDIR="${ROOT}/logs"

#### Options ####
rm -R -f "$ROOT"
mkdir "$ROOT"
mkdir "$LOGDIR"
mkdir "$ROOTDATA"

# Download & preprocess the data.
mkdir "${ROOTRAWDATA}"
pushd "$ROOT"
wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip training.1600000.processed.noemoticon.csv
mv training.1600000.processed.noemoticon.csv "${ROOTRAWDATA}"
rm trainingandtestdata.zip
popd
$BAZEL run -c opt --copt=-mavx2 :preprocess_sentiment140

# Run the experiments. This takes time (>6h on a single machine).
training_data="${ROOTRAWDATA}/train.csv"
test_data="${ROOTRAWDATA}/test.csv"
dictionary="${ROOTRAWDATA}/dict.txt"

batch_size=128
num_days=15
vocab_size=4096
test_after=0
bow_limit=4
num_groups=4
num_train_examples_per_day=144000

# A simple version of the setting
learning_rates=(0.1)
modes=('time-feature' 'fourier' 'pluralistic' 'online-learning')
biases=(0.7)
replicas=(0 1 2 3 4)
fourier_dim=(64)

task_id=0
for learning_rate in ${learning_rates[@]}; do
  for mode in ${modes[@]}; do
    for bias in ${biases[@]}; do
      for replica in ${replicas[@]}; do
        for fourier_dim in ${fourier_dim[@]}; do
          echo "Running task $task_id"
          $BAZEL run -c opt --copt=-mavx2 :cyclic_bag_log_reg_online_learning --   \
              --task_id=$task_id                                       \
              --log_file="${LOGDIR}/${task_id}.log"                    \
              --training_data=$training_data                           \
              --test_data=$test_data                                   \
              --dictionary=$dictionary                                 \
              --lr=$learning_rate                                      \
              --batch_size=$batch_size                                 \
              --num_days=$num_days                                     \
              --vocab_size=$vocab_size                                 \
              --bow_limit=$bow_limit                                   \
              --test_after=$test_after                                 \
              --mode=$mode                                             \
              --num_groups=$num_groups                                 \
              --num_train_examples_per_day=$num_train_examples_per_day \
              --bias=$bias                                             \
              --replica=$replica                                       \
              --fourier_dim=$fourier_dim
          task_id=$((task_id+1))
        done
      done
    done
  done
done
