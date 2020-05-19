# Copyright 2020 Google LLC
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

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import numpy as np

# A workaround to avoid crash because tfds may open to many files.
import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000


DATASET_SPLITS = {
  'cifar10': {'train': 'train[:98%]', 'test': 'test'},
  'cifar100': {'train': 'train[:98%]', 'test': 'test'},
  'imagenet2012': {'train': 'train[:99%]', 'test': 'validation'},
}


def get_dataset_info(dataset, split, examples_per_class):
  data_builder = tfds.builder(dataset)
  original_num_examples = data_builder.info.splits[split].num_examples
  num_classes = data_builder.info.features['label'].num_classes
  if examples_per_class is not None:
    num_examples = examples_per_class * num_classes
  else:
    num_examples = original_num_examples
  return {'original_num_examples': original_num_examples,
          'num_examples': num_examples,
          'num_classes': num_classes}


def sample_subset(data, num_examples, num_classes,
                  examples_per_class, examples_per_class_seed):
  data = data.batch(min(num_examples, MAX_IN_MEMORY))

  data = data.as_numpy_iterator().next()

  np.random.seed(examples_per_class_seed)
  indices = [idx
             for c in range(num_classes)
             for idx in np.random.choice(np.where(data['label'] == c)[0],
                                         examples_per_class,
                                         replace=False)]

  data = {'image': data['image'][indices],
          'label': data['label'][indices]}

  data = tf.data.Dataset.zip(
    (tf.data.Dataset.from_tensor_slices(data['image']),
     tf.data.Dataset.from_tensor_slices(data['label'])))
  return data.map(lambda x, y: {'image': x, 'label': y},
                  tf.data.experimental.AUTOTUNE)


def get_data(dataset, mode,
             repeats, batch_size,
             resize_size, crop_size,
             mixup_alpha,
             examples_per_class, examples_per_class_seed,
             num_devices,
             tfds_manual_dir):

  split = DATASET_SPLITS[dataset][mode]
  dataset_info = get_dataset_info(dataset, split, examples_per_class)

  data_builder = tfds.builder(dataset)
  data_builder.download_and_prepare(
   download_config=tfds.download.DownloadConfig(manual_dir=tfds_manual_dir))
  data = data_builder.as_dataset(
    split=split,
    decoders={'image': tfds.decode.SkipDecoding()})
  decoder = data_builder.info.features['image'].decode_example

  if (mode == 'train') and (examples_per_class is not None):
    data = sample_subset(data,
                         dataset_info['original_num_examples'],
                         dataset_info['num_classes'],
                         examples_per_class, examples_per_class_seed)

  def _pp(data):
    im = decoder(data['image'])
    if mode == 'train':
      im = tf.image.resize(im, [resize_size, resize_size])
      im = tf.image.random_crop(im, [crop_size, crop_size, 3])
      im = tf.image.flip_left_right(im)
    else:
      # usage of crop_size here is intentional
      im = tf.image.resize(im, [crop_size, crop_size])
    im = (im - 127.5) / 127.5
    label = tf.one_hot(data['label'], dataset_info['num_classes'])
    return {'image': im, 'label': label}

  data = data.cache()
  data = data.repeat(repeats)
  if mode == 'train':
    data = data.shuffle(min(dataset_info['num_examples'], MAX_IN_MEMORY))
  data = data.map(_pp, tf.data.experimental.AUTOTUNE)
  data = data.batch(batch_size, drop_remainder=True)

  def _mixup(data):
    beta_dist = tfp.distributions.Beta(mixup_alpha, mixup_alpha)
    beta = tf.cast(beta_dist.sample([]), tf.float32)
    data['image'] = (beta * data['image'] +
                     (1 - beta) * tf.reverse(data['image'], axis=[0]))
    data['label'] = (beta * data['label'] +
                     (1 - beta) * tf.reverse(data['label'], axis=[0]))
    return data

  if mixup_alpha is not None and mixup_alpha > 0.0 and mode == 'train':
    data = data.map(_mixup, tf.data.experimental.AUTOTUNE)

  # Shard data such that it can be distributed accross devices
  def _shard(data):
    data['image'] = tf.reshape(data['image'],
                               [num_devices, -1, crop_size, crop_size, 3])
    data['label'] = tf.reshape(data['label'],
                               [num_devices, -1, dataset_info['num_classes']])
    return data
  if num_devices is not None:
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)

  return data.prefetch(1)
