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

Modification adds Fourier learning method into the model structure, and prunes the code irrelavent to Fourier learning paper's experiment settings.

"""

import csv, datetime, os, random, sys, time, pathlib

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

print(str(pathlib.Path(__file__).parent.parent.resolve())+"/")
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve())+"/")

from semi_cyclic_sgd import sentiment_util as su

FLAGS = flags.FLAGS
RAWDATA = '/tmp/sc_paper/raw_data/'
flags.DEFINE_integer('task_id', -1,
                     'Task id specifying which experiment to run.')
flags.DEFINE_string('log_file', '/tmp/sc.log', 'Log file path.')
flags.DEFINE_string('training_data', os.path.join(RAWDATA, 'train.csv'),
                    'Path to training data file.')
flags.DEFINE_string('test_data', os.path.join(RAWDATA, 'test.csv'),
                    'Path to test data file.')
flags.DEFINE_string(
    'dictionary', os.path.join(RAWDATA, 'dict.txt'),
    'Dictionary file (one word per line, descending by '
    'frequency).')
flags.DEFINE_float('lr', 0.215, 'Learning rate.')
flags.DEFINE_integer('batch_size', 128, 'Minibatch size.')
flags.DEFINE_integer('num_days', 10, 'Number of days to train')
flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size (0: unlimited).')
flags.DEFINE_integer('bow_limit', 0,
                     'Max num of words in bow presentation (0: unlimited).')
flags.DEFINE_integer(
    'test_after', 0, 'compute test data metrics every test_after training'
    ' examples. If 0, don\'t.')
flags.DEFINE_string(
    'mode', 'iid',
    'Train/test mode. One of iid, sep (train separate models), sc '
    '(semi-cyclic + pluralistic)')
flags.DEFINE_integer(
    'num_groups', 2, 'Number of groups the data will be split up into. '
    'Every day, num_groups blocks of data are created, each '
    'with num_train_examples_per_day/num_groups examples.')
flags.DEFINE_integer(
    'num_train_examples_per_day', 144000,
    'Number of examples per day for training. Default is picked such that '
    'for a 90/10 train/test split, ten days corresponds to '
    'one training pass over the complete dataset (which is sufficient '
    'for convergence).')
flags.DEFINE_float('bias', 0.0,
                   'Peak percentage of pos or neg examples to drop')
flags.DEFINE_integer(
    'replica', 0, 'Replica - this script may be invoked multiple times with '
    'identical parameters, to compute means+stds. This is the one '
    'flag that varies, and used as a seed')
# Fourier Learning Parameters
flags.DEFINE_integer(
    'fourier_dim', 16, 'Fourier hidden layer dimension - this is the dimension of the hidden layer size used to transform the neural network into the fourier domain.'
)
flags.DEFINE_integer(
    'T', 86400, 'Period of the data distribution.'
)
flags.DEFINE_integer(
    'hidden_layer_dim', 64, 'Dimension of last hidden layer.'
)

class Model(object):
  """Class representing a logistic regression model using bag-of-words."""

  def __init__(self, lr, vocab, bow_limit, is_fourier, use_time_feature):
    self.vocab = vocab
    self.input_dim = len(self.vocab)
    self.lr = lr
    self.num_classes = 2
    self.bow_limit = bow_limit
    self._optimizer = None
    # Fourier analysis parameters
    self._is_fourier = is_fourier
    self._use_time_feature = use_time_feature 
    self._fourier_dim = FLAGS.fourier_dim
    self._T = FLAGS.T
    self._layer_dim = FLAGS.hidden_layer_dim

  @property
  def optimizer(self):
    """Optimizer to be used by the model."""
    if self._optimizer is None:
      self._optimizer = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate=self.lr)
    return self._optimizer

  def create_model(self):
    """Creates a TF model and returns ops necessary to run training/eval."""
    """ Version 2 """
    features = tf.compat.v1.placeholder(tf.float32, [None, self.input_dim])
    labels = tf.compat.v1.placeholder(tf.float32, [None, self.num_classes])
    timestamp = tf.compat.v1.placeholder(tf.int64, [None, 1])
    t_of_day = tf.cast(tf.mod(timestamp, self._T), tf.float32)
    
    if not self._use_time_feature:
      # MLP to last hidden layer
      W = tf.Variable(tf.random.normal(shape=[self.input_dim, self._layer_dim], stddev=0.1))
      B = tf.Variable(tf.random.normal(shape=[self._layer_dim], stddev=0.1))
      input = features
      last_hidden_layer = tf.nn.relu(tf.matmul(input, W) + B)
      # Normal Output Layer
      W_output = tf.Variable(tf.random.normal(shape=[self._layer_dim, self.num_classes], stddev=0.1))
      B_output = tf.Variable(tf.random.normal(shape=[self.num_classes], stddev=0.1))
      normal_logit = tf.matmul(last_hidden_layer, W_output) + B_output
      # Fourier Output Layer
      W_fourier_output = tf.Variable(tf.random.normal(shape=[2 * self._fourier_dim, self.num_classes], stddev=0.1))
      B_fourier_output = tf.Variable(tf.random.normal(shape=[self.num_classes], stddev=0.1))
      W_re = tf.Variable(tf.random.normal(shape=[self._layer_dim, self._fourier_dim], stddev=0.1))
      B_re = tf.Variable(tf.random.normal(shape=[self._fourier_dim], stddev=0.1))
      W_im = tf.Variable(tf.random.normal(shape=[self._layer_dim, self._fourier_dim], stddev=0.1))
      B_im = tf.Variable(tf.random.normal(shape=[self._fourier_dim], stddev=0.1))
      fourier_layer_re = tf.nn.relu(tf.matmul(last_hidden_layer, W_re) + B_re) * tf.concat([tf.cos(n * t_of_day / self._T) for n in range(self._fourier_dim)], axis=-1)
      fourier_layer_im = tf.nn.relu(tf.matmul(last_hidden_layer, W_im) + B_im) * tf.concat([tf.sin(n * t_of_day / self._T) for n in range(self._fourier_dim)], axis=-1)
      fourier_logit = tf.matmul(tf.concat([fourier_layer_re, fourier_layer_im], axis=-1), W_fourier_output) + B_fourier_output
    else:
      # MLP to last hidden layer
      W = tf.Variable(tf.random.normal(shape=[self.input_dim+1, self._layer_dim], stddev=0.1))
      B = tf.Variable(tf.random.normal(shape=[self._layer_dim], stddev=0.1))
      input = tf.concat([features, t_of_day / self._T], axis=-1)
      last_hidden_layer = tf.nn.relu(tf.matmul(input, W) + B)
      # Normal Output Layer
      W_output = tf.Variable(tf.random.normal(shape=[self._layer_dim, self.num_classes], stddev=0.1))
      B_output = tf.Variable(tf.random.normal(shape=[self.num_classes], stddev=0.1))
      normal_logit = tf.matmul(last_hidden_layer, W_output) + B_output
      # Fourier Output Layer
      W_fourier_output = tf.Variable(tf.random.normal(shape=[2 * self._fourier_dim, self.num_classes], stddev=0.1))
      B_fourier_output = tf.Variable(tf.random.normal(shape=[self.num_classes], stddev=0.1))
      W_re = tf.Variable(tf.random.normal(shape=[self._layer_dim, self._fourier_dim], stddev=0.1))
      B_re = tf.Variable(tf.random.normal(shape=[self._fourier_dim], stddev=0.1))
      W_im = tf.Variable(tf.random.normal(shape=[self._layer_dim, self._fourier_dim], stddev=0.1))
      B_im = tf.Variable(tf.random.normal(shape=[self._fourier_dim], stddev=0.1))
      fourier_layer_re = tf.nn.relu(tf.matmul(last_hidden_layer, W_re) + B_re) * tf.concat([tf.cos(n * t_of_day / self._T) for n in range(self._fourier_dim)], axis=-1)
      fourier_layer_im = tf.nn.relu(tf.matmul(last_hidden_layer, W_im) + B_im) * tf.concat([tf.sin(n * t_of_day / self._T) for n in range(self._fourier_dim)], axis=-1)
      fourier_logit = tf.matmul(tf.concat([fourier_layer_re, fourier_layer_im], axis=-1), W_fourier_output) + B_fourier_output

    if not self._is_fourier:
      pred = tf.nn.softmax(normal_logit)
    else:
      pred = tf.nn.softmax(fourier_logit)

    loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(pred), axis=1))
    train_op = self.optimizer.minimize(
        loss=loss, global_step=tf.train.get_or_create_global_step())

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    eval_metric_op = tf.count_nonzero(correct_pred)

    return features, labels, train_op, loss, eval_metric_op, timestamp

  def process_x(self, raw_batch):
    x_batch = [e[0][4] for e in raw_batch]  # list of lines/phrases
    bags = np.zeros((len(raw_batch), len(self.vocab)))
    su.bag_of_words(x_batch, bags, self.bow_limit)
    return bags

  def process_y(self, raw_batch):
    y_batch = [int(e[1]) for e in raw_batch]
    y_batch = [su.val_to_vec(self.num_classes, e) for e in y_batch]
    y_batch = np.array(y_batch)
    return y_batch
  
  def process_t(self, raw_batch):
    t_batch = np.array([e[0][5] for e in raw_batch]).reshape(-1, 1).astype(np.int64)
    return t_batch

class CyclicDataGenerator(object):
  """Generates minibatches from data grouped by day & group.

  The class does not handle loading or preprocessing (such as shuffling) of
  data, and subclasses should implement that in their constructor.
  """

  def __init__(self, logger, num_groups, num_examples_per_day_per_group,
               batch_size):
    # self.data has num_groups list. Each contains the entire data of that
    # group, which is processed in a round-robin manner, see comments on get().
    assert batch_size != 0
    self.logger = logger
    self.num_groups = num_groups
    self.data = [[] for _ in range(0, num_groups)]  # can't use [[]]*num_groups
    self.num_examples_per_day_per_group = num_examples_per_day_per_group
    self.batch_size = batch_size

  def get(self, day, group):
    """Gets data for training - a generator representing one block."""
    # Compute where inside the group to start. We assume round-robin processing,
    # e.g. for a group of size 10, 2 groups, 8 examples per day, we'd have 4
    # examples per day per group, so on day 1 we'd return examples 0..3, on day
    # 2 examples 4..7, on day 3 examples 8,9,0,1, etc. That also works when we
    # have to iterate through a group more than once per day.
    assert group < self.num_groups
    start_index = day * self.num_examples_per_day_per_group
    end_index = (day + 1) * self.num_examples_per_day_per_group

    for i in range(start_index, end_index, self.batch_size):
      group_size = len(self.data[group])
      assert group_size >= self.batch_size
      i_mod = i % group_size
      if i_mod + self.batch_size <= group_size:
        result = self.data[group][i_mod:i_mod + self.batch_size]
        if not result:
          print(day, group, self.num_groups, start_index, end_index, i,
                group_size, i_mod, self.batch_size)
      else:
        result = self.data[group][i_mod:]
        remainder = self.batch_size - len(result)
        result.extend(self.data[group][:remainder])
        if not result:
          print(day, group, self.num_groups, start_index, end_index, i,
                group_size, i_mod, self.batch_size)
      yield result

  def get_test_data(self, group):
    # Gets all data for a group, ignoring num_examples_per_day. Used for test
    # data where splitting into days may be undesired.
    assert group < self.num_groups
    for i in range(0,
                   len(self.data[group]) - self.batch_size + 1,
                   self.batch_size):
      yield self.data[group][i:i + self.batch_size]

  def process_row(self, row, vocab):
    """Utility function that preprocesses a Sentiment140 training example.

    Args:
      row: a row, split into items, from the Sentiment140 CSV file.
      vocab: a vocabulary for tokenization.  The example is processed by
        converting the label to an integer, and tokenizing the text of the post.
        The argument row is modified in place.
    """
    if row[0] == '1':
      row[0] = 1
    elif row[0] == '0':
      row[0] = 0
    else:
      raise ValueError('label neither 0 nor 1, but: type {}, value {}'.format(
          type(row[0]), row[0]))
    row[5] = su.line_to_word_ids(row[5], vocab)
    # Yingxiang processing timestamp
    # time_obj = parser.parse(row[2]) # ignoring timezone
    time_obj = datetime.datetime.strptime(row[2].replace("PDT", "GMT"), '%a %b %d %H:%M:%S %Z %Y')
    time_stamp = np.mod(time_obj.timestamp(), FLAGS.T) / 3600
    if len(row) < 7:
      row.append(time_stamp) # row[6]
    else:
      row[6] = time_stamp

class NonIidDataGenerator(CyclicDataGenerator):
  """A data generator for block cyclic data."""

  def __init__(self,
               logger,
               path,
               vocab,
               num_groups,
               num_examples_per_day_per_group,
               bias,
               batch_size=0):
    CyclicDataGenerator.__init__(self, logger, num_groups,
                                 num_examples_per_day_per_group, batch_size)
    # Bias parameter b=0..1 specifies how much to bias. In group 0, we drop
    # b*100% of the negative examples - let's call this biasing by +b.
    # In group num_groups/2+1, we drop b*100% of the positive
    # examples - say, biasing by -b. And then we go back to +b again, to have
    # some continuity. That interpolation only works cleanly with an even
    # num_groups.
    true_num_groups = 8 # Hard-wire the number of groups to 12 to create a continuously oscillating data distribution as much as the original code allows.
    assert true_num_groups % 2 == 0
    repeats = 2
    # Python 2: type(x/2) == int, Python 3: type(x/2) == float.
    biases = np.interp(
        range(true_num_groups // 2 + 1), [0, true_num_groups / 2],
        [-bias, bias]).tolist()
    biases.extend(biases[-2:0:-1])
    biases *= repeats
    random.seed(0)
    random.shuffle(biases)
    true_num_groups *= repeats
    with open(path, 'r') as f:
      csv_reader = csv.reader(f, delimiter=',')
      for row in csv_reader:
        self.process_row(row, vocab)
        label = row[0]
        t = datetime.datetime.strptime(row[2].replace("PDT", "GMT"), '%a %b %d %H:%M:%S %Z %Y')
        # if len(row) < 7:
        #   row.append(t.timestamp())
        # else:
        #   row[6] = t.timestamp()  
        # 1. Split by time of day into num_groups.
        assert 24 % num_groups == 0
        group = int(t.hour / (24 / num_groups))
        true_group = int(t.hour / (24 / true_num_groups))
        # 2. Introduce further bias by dropping examples.
        if bias > 0.0:
          r = random.random()
          b = biases[true_group]
          # Drop neg?
          if b < 0:
            if label == 1 or r >= abs(b):
              self.data[group].append([row[1:], label])
          else:
            if label == 0 or r >= b:
              self.data[group].append([row[1:], label])
        else:
          # No biasing, add unconditionally.
          self.data[group].append([row[1:], label])
    for g in range(0, self.num_groups):
      logger.log('group {}: {} examples'.format(g, len(self.data[g])))
      random.shuffle(self.data[g])
      self.data[g].sort(key=lambda x: x[0][5]) # sort according to time stamp
      print(self.data[g][0:10])


class Logger(object):
  """A logger that logs to stdout and to a logfile, with throttling."""

  def __init__(self, interval=1):
    self.t = 0
    self.interval = interval
    self.out_file = open(FLAGS.log_file, 'w')

  def maybe_log(self, message):
    """Log if the last call to maybe_log was more than interval seconds ago."""
    cur_time = time.time()
    if self.t == 0 or self.t + self.interval < cur_time:
      print(message)
      print(message, file=self.out_file)
      self.t = cur_time

  def log(self, message):
    print(message)
    print(message, file=self.out_file)


def log_config(logger):
  """Logs the configuration of this run, so it can be used in the analysis phase."""
  logger.log('== Configuration ==')
  logger.log('task_id=%d' % FLAGS.task_id)
  logger.log('lr=%f' % FLAGS.lr)
  logger.log('vocab_size=%s' % FLAGS.vocab_size)
  logger.log('batch_size=%s' % FLAGS.batch_size)
  logger.log('bow_limit=%s' % FLAGS.bow_limit)
  logger.log('training_data=%s' % FLAGS.training_data)
  logger.log('test_data=%s' % FLAGS.test_data)
  logger.log('num_groups=%d' % FLAGS.num_groups)
  logger.log('num_days=%d' % FLAGS.num_days)
  logger.log('num_train_examples_per_day=%d' % FLAGS.num_train_examples_per_day)
  logger.log('mode=%s' % FLAGS.mode)
  logger.log('bias=%f' % FLAGS.bias)
  logger.log('replica=%d' % FLAGS.replica)
  logger.log('fourier_dim=%d' % FLAGS.fourier_dim)


def test(test_data,
         model,
         sess,
         eval_metric_op,
         features,
         labels,
         timestamp,
         logger,
         d,
         g,
         num_train_examples,
         prefix='',
         skip=True):
  """Tests the current model on all the data from test_data for group g."""
  cur_time = time.time()
  num_correct = 0
  num_test_examples = 0
  t_process_x = 0
  t_process_y = 0
  t_process_t = 0
  t_process_tf = 0
  if skip is True:
    logger.log(
        '%sday %d, group %g: num_train_examples %d (dt=%ds): num correct: %d/%d (%f)'
        % (prefix, d, g, num_train_examples, 0, 0, 1,
          0.0))
  else:
    for b in test_data.get_test_data(g):
    # for b in test_data.get(d, g):
      t1 = time.clock()
      x = model.process_x(b)
      t2 = time.clock()
      y = model.process_y(b)
      t3 = time.clock()
      ts = model.process_t(b)
      t4 = time.clock()
      num_test_examples = num_test_examples + len(x)
      num_correct_ = sess.run([eval_metric_op], {features: x, labels: y, timestamp: ts})
      t5 = time.clock()
      num_correct = num_correct + num_correct_[0]
      t_process_x = t_process_x + (t2 - t1)
      t_process_y = t_process_y + (t3 - t2)
      t_process_t = t_process_t + (t4 - t3)
      t_process_tf = t_process_tf + (t5 - t4)
    dt = time.time() - cur_time
    logger.log(
        '%sday %d, group %g: num_train_examples %d (dt=%ds): num correct: %d/%d (%f)'
        % (prefix, d, g, num_train_examples, dt, num_correct, num_test_examples,
          num_correct / float(num_test_examples)))


def init(logger):
  """Loads + groups data, dictionary."""
  vocab = {}
  i = 0
  with open(FLAGS.dictionary) as f:
    for l in f:
      w = l.strip()
      vocab[w] = i
      i = i + 1
      if FLAGS.vocab_size > 0 and i >= FLAGS.vocab_size:
        break
  logger.log('Read vocabulary with %d words' % len(vocab))
  logger.log('Loading training & testing data')
  training_data = NonIidDataGenerator(
      logger, FLAGS.training_data, vocab, FLAGS.num_groups,
      int(FLAGS.num_train_examples_per_day / FLAGS.num_groups), FLAGS.bias,
      FLAGS.batch_size)
  test_data = NonIidDataGenerator(logger, FLAGS.test_data, vocab,
                                  FLAGS.num_groups, 0, FLAGS.bias,
                                  FLAGS.batch_size)
  return vocab, training_data, test_data


def main(unused_args):
  logger = Logger(10)
  log_config(logger)
  vocab, training_data, test_data = init(logger)
  # test_data = training_data # short-circuited test data to be training data.
  logger.log('Creating model(s)')
  tf.set_random_seed(FLAGS.replica)
  models = []
  if FLAGS.mode == 'pluralistic':
    num_models = FLAGS.num_groups
    FLAGS.hidden_layer_dim = int(FLAGS.hidden_layer_dim / FLAGS.num_groups)
  else:
    num_models = 1
    # FLAGS.fourier_dim = FLAGS.fourier_dim * FLAGS.num_groups
  for _ in range(num_models):
    is_fourier = True if FLAGS.mode == 'fourier' else False
    use_time_feature = True if FLAGS.mode == 'time-feature' else False
    model = Model(FLAGS.lr, vocab, FLAGS.bow_limit, is_fourier, use_time_feature)
    features, labels, train_op, loss_op, eval_metric_op, timestamp = model.create_model()
    models.append({
        'model': model,
        'features': features,
        'labels': labels,
        'timestamp': timestamp,
        'train_op': train_op,
        'loss_op': loss_op,
        'eval_metric_op': eval_metric_op
    })

  with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for d in range(0, FLAGS.num_days):
      num_train_examples = 0
      for g in range(0, FLAGS.num_groups):
        m = models[0] if FLAGS.mode != 'pluralistic' else models[g]
        # Test
        if FLAGS.mode == 'online-learning':
          for gt in range(0, FLAGS.num_groups):
            skip = False if gt == g else True
            test(
                test_data,
                m['model'],
                sess,
                m['eval_metric_op'],
                m['features'],
                m['labels'],
                m['timestamp'],
                logger,
                d,
                gt,
                num_train_examples,
                prefix='oco %d on %d: ' % (g, gt),
                skip=skip)
        elif FLAGS.mode == 'pluralistic':
          for gt in range(0, FLAGS.num_groups):
            skip = False if gt == g else True
            test(
                test_data,
                m['model'],
                sess,
                m['eval_metric_op'],
                m['features'],
                m['labels'],
                m['timestamp'],
                logger,
                d,
                gt,
                num_train_examples,
                prefix='sc %d on %d: ' % (g, gt),
                skip=skip)
        elif FLAGS.mode == 'fourier':
          for gt in range(0, FLAGS.num_groups):
            skip = False if gt == g else True
            test(
            test_data,
            m['model'],
            sess,
            m['eval_metric_op'],
            m['features'],
            m['labels'],
            m['timestamp'],
            logger,
            d,
            gt,
            num_train_examples,
            prefix='fourier %d on %d: ' % (g, gt),
            skip=skip)
        elif FLAGS.mode == 'time-feature':
          for gt in range(0, FLAGS.num_groups):
            skip = False if gt == g else True
            test(
            test_data,
            m['model'],
            sess,
            m['eval_metric_op'],
            m['features'],
            m['labels'],
            m['timestamp'],
            logger,
            d,
            gt,
            num_train_examples,
            prefix='time %d on %d: ' % (g, gt),
            skip=skip)
        else:
          raise ValueError('unsupported mode %s' % FLAGS.mode)
        # Train.
        num_train_examples = 0
        for batch in training_data.get(d, g):
          x = m['model'].process_x(batch)
          y = m['model'].process_y(batch)
          t = m['model'].process_t(batch)
          num_train_examples = num_train_examples + len(x)
          # loss = sess.run([m['loss_op']], {
          #     m['features']: x,
          #     m['labels']: y,
          #     m['timestamp']: t
          # })
          loss, _ = sess.run([m['loss_op'], m['train_op']], {
              m['features']: x,
              m['labels']: y,
              m['timestamp']: t
          })
          logger.maybe_log('day %d, group %d: trained on %d examples, loss=%f' %
                           (d, g, num_train_examples, loss))

  logger.log('END_MARKER')


if __name__ == '__main__':
  if not sys.version_info >= (3, 0):
    print('This script requires Python3 to run.')
    sys.exit(1)
  app.run(main)
