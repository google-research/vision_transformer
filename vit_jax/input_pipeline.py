# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

from absl import logging
import flax
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import sys
if sys.platform != 'darwin':
  # A workaround to avoid crash because tfds may open to many files.
  import resource
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000


def get_tfds_info(dataset, split):
  """Returns information about tfds dataset -- see `get_dataset_info()`."""
  data_builder = tfds.builder(dataset)
  return dict(
      num_examples=data_builder.info.splits[split].num_examples,
      num_classes=data_builder.info.features['label'].num_classes,
      int2str=data_builder.info.features['label'].int2str,
      examples_glob=None,
  )


def get_directory_info(directory):
  """Returns information about directory dataset -- see `get_dataset_info()`."""
  examples_glob = f'{directory}/*/*.jpg'
  paths = glob.glob(examples_glob)
  get_classname = lambda path: path.split('/')[-2]
  class_names = sorted(set(map(get_classname, paths)))
  return dict(
      num_examples=len(paths),
      num_classes=len(class_names),
      int2str=lambda id_: class_names[id_],
      examples_glob=examples_glob,
  )


def get_dataset_info(dataset, split):
  """Returns information about a dataset.

  Args:
    dataset: Name of tfds dataset or directory -- see `./configs/common.py`
    split: Which split to return data for (e.g. "test", or "train"; tfds also
      supports splits like "test[:90%]").

  Returns:
    A dictionary with the following keys:
    - num_examples: Number of examples in dataset/mode.
    - num_classes: Number of classes in dataset.
    - int2str: Function converting class id to class name.
    - examples_glob: Glob to select all files, or None (for tfds dataset).
  """
  directory = os.path.join(dataset, split)
  if os.path.isdir(directory):
    return get_directory_info(directory)
  return get_tfds_info(dataset, split)


def get_datasets(config):
  """Returns `ds_train, ds_test` for specified `config`."""

  if os.path.isdir(config.dataset):
    train_dir = os.path.join(config.dataset, 'train')
    test_dir = os.path.join(config.dataset, 'test')
    if not os.path.isdir(train_dir):
      raise ValueError('Expected to find directories"{}" and "{}"'.format(
          train_dir,
          test_dir,
      ))
    logging.info('Reading dataset from directories "%s" and "%s"', train_dir,
                 test_dir)
    ds_train = get_data_from_directory(
        config=config, directory=train_dir, mode='train')
    ds_test = get_data_from_directory(
        config=config, directory=test_dir, mode='test')
  else:
    logging.info('Reading dataset from tfds "%s"', config.dataset)
    ds_train = get_data_from_tfds(config=config, mode='train')
    ds_test = get_data_from_tfds(config=config, mode='test')

  return ds_train, ds_test


def get_data_from_directory(*, config, directory, mode):
  """Returns dataset as read from specified `directory`."""

  dataset_info = get_directory_info(directory)
  data = tf.data.Dataset.list_files(dataset_info['examples_glob'])
  class_names = [
      dataset_info['int2str'](id_) for id_ in range(dataset_info['num_classes'])
  ]

  def _pp(path):
    return dict(
        image=path,
        label=tf.where(
            tf.strings.split(path, '/')[-2] == class_names
        )[0][0],
    )

  image_decoder = lambda path: tf.image.decode_jpeg(tf.io.read_file(path), 3)

  return get_data(
      data=data,
      mode=mode,
      num_classes=dataset_info['num_classes'],
      image_decoder=image_decoder,
      repeats=None if mode == 'train' else 1,
      batch_size=config.batch_eval if mode == 'test' else config.batch,
      image_size=config.pp['crop'],
      shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer),
      preprocess=_pp)


def get_data_from_tfds(*, config, mode):
  """Returns dataset as read from tfds dataset `config.dataset`."""

  data_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)

  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(
          manual_dir=config.tfds_manual_dir))
  data = data_builder.as_dataset(
      split=config.pp[mode],
      # Reduces memory footprint in shuffle buffer.
      decoders={'image': tfds.decode.SkipDecoding()},
      shuffle_files=mode == 'train')
  image_decoder = data_builder.info.features['image'].decode_example

  dataset_info = get_tfds_info(config.dataset, config.pp[mode])
  return get_data(
      data=data,
      mode=mode,
      num_classes=dataset_info['num_classes'],
      image_decoder=image_decoder,
      repeats=None if mode == 'train' else 1,
      batch_size=config.batch_eval if mode == 'test' else config.batch,
      image_size=config.pp['crop'],
      shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer))


def get_data(*,
             data,
             mode,
             num_classes,
             image_decoder,
             repeats,
             batch_size,
             image_size,
             shuffle_buffer,
             preprocess=None):
  """Returns dataset for training/eval.

  Args:
    data: tf.data.Dataset to read data from.
    mode: Must be "train" or "test".
    num_classes: Number of classes (used for one-hot encoding).
    image_decoder: Applied to `features['image']` after shuffling. Decoding the
      image after shuffling allows for a larger shuffle buffer.
    repeats: How many times the dataset should be repeated. For indefinite
      repeats specify None.
    batch_size: Global batch size. Note that the returned dataset will have
      dimensions [local_devices, batch_size / local_devices, ...].
    image_size: Image size after cropping (for training) / resizing (for
      evaluation).
    shuffle_buffer: Number of elements to preload the shuffle buffer with.
    preprocess: Optional preprocess function. This function will be applied to
      the dataset just after repeat/shuffling, and before the data augmentation
      preprocess step is applied.
  """

  def _pp(data):
    im = image_decoder(data['image'])
    if mode == 'train':
      channels = im.shape[-1]
      begin, size, _ = tf.image.sample_distorted_bounding_box(
          tf.shape(im),
          tf.zeros([0, 0, 4], tf.float32),
          area_range=(0.05, 1.0),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)
      im = tf.slice(im, begin, size)
      # Unfortunately, the above operation loses the depth-dimension. So we
      # need to restore it the manual way.
      im.set_shape([None, None, channels])
      im = tf.image.resize(im, [image_size, image_size])
      if tf.random.uniform(shape=[]) > 0.5:
        im = tf.image.flip_left_right(im)
    else:
      im = tf.image.resize(im, [image_size, image_size])
    im = (im - 127.5) / 127.5
    label = tf.one_hot(data['label'], num_classes)  # pylint: disable=no-value-for-parameter
    return {'image': im, 'label': label}

  data = data.repeat(repeats)
  if mode == 'train':
    data = data.shuffle(shuffle_buffer)
  if preprocess is not None:
    data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
  data = data.map(_pp, tf.data.experimental.AUTOTUNE)
  data = data.batch(batch_size, drop_remainder=True)

  # Shard data such that it can be distributed accross devices
  num_devices = jax.local_device_count()

  def _shard(data):
    data['image'] = tf.reshape(data['image'],
                               [num_devices, -1, image_size, image_size,
                                data['image'].shape[-1]])
    data['label'] = tf.reshape(data['label'],
                               [num_devices, -1, num_classes])
    return data

  if num_devices is not None:
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)

  return data.prefetch(1)


def prefetch(dataset, n_prefetch):
  """Prefetches data to device and converts to numpy array."""
  ds_iter = iter(dataset)
  ds_iter = map(lambda x: jax.tree.map(lambda t: np.asarray(memoryview(t)), x),
                ds_iter)
  if n_prefetch:
    ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
  return ds_iter
