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

# Lint as: python3
# coding: utf-8

from functools import partial
import time
import os

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import bit_common
import bit_hyperrule
import bit_tf2.models as models
import input_pipeline_tf2_or_jax as input_pipeline


def reshape_for_keras(features, batch_size, crop_size):
  features["image"] = tf.reshape(features["image"], (batch_size, crop_size, crop_size, 3))
  features["label"] = tf.reshape(features["label"], (batch_size, -1))
  return (features["image"], features["label"])


class BiTLRSched(tf.keras.callbacks.Callback):
  def __init__(self, base_lr, num_samples):
    self.step = 0
    self.base_lr = base_lr
    self.num_samples = num_samples

  def on_train_batch_begin(self, batch, logs=None):
    lr = bit_hyperrule.get_lr(self.step, self.num_samples, self.base_lr)
    tf.keras.backend.set_value(self.model.optimizer.lr, lr)
    self.step += 1



def main(args):
  tf.io.gfile.makedirs(args.logdir)
  logger = bit_common.setup_logger(args)

  logger.info(f'Available devices: {tf.config.list_physical_devices()}')

  tf.io.gfile.makedirs(args.bit_pretrained_dir)
  bit_model_file = os.path.join(args.bit_pretrained_dir, f'{args.model}.h5')
  if not tf.io.gfile.exists(bit_model_file):
    model_url = models.KNOWN_MODELS[args.model]
    logger.info(f'Downloading the model from {model_url}...')
    tf.io.gfile.copy(model_url, bit_model_file)

  # Set up input pipeline
  dataset_info = input_pipeline.get_dataset_info(
    args.dataset, 'train', args.examples_per_class)

  # Distribute training
  strategy = tf.distribute.MirroredStrategy()
  num_devices = strategy.num_replicas_in_sync
  print('Number of devices: {}'.format(num_devices))

  resize_size, crop_size = bit_hyperrule.get_resolution_from_dataset(args.dataset)
  data_train = input_pipeline.get_data(
    dataset=args.dataset, mode='train',
    repeats=None, batch_size=args.batch,
    resize_size=resize_size, crop_size=crop_size,
    examples_per_class=args.examples_per_class,
    examples_per_class_seed=args.examples_per_class_seed,
    mixup_alpha=bit_hyperrule.get_mixup(dataset_info['num_examples']),
    num_devices=num_devices,
    tfds_manual_dir=args.tfds_manual_dir)
  data_test = input_pipeline.get_data(
    dataset=args.dataset, mode='test',
    repeats=1, batch_size=args.batch,
    resize_size=resize_size, crop_size=crop_size,
    examples_per_class=1, examples_per_class_seed=0,
    mixup_alpha=None,
    num_devices=num_devices,
    tfds_manual_dir=args.tfds_manual_dir)

  data_train = data_train.map(lambda x: reshape_for_keras(
    x, batch_size=args.batch, crop_size=crop_size))
  data_test = data_test.map(lambda x: reshape_for_keras(
    x, batch_size=args.batch, crop_size=crop_size))

  with strategy.scope():
    filters_factor = int(args.model[-1])*4
    model = models.ResnetV2(
        num_units=models.NUM_UNITS[args.model],
        num_outputs=21843,
        filters_factor=filters_factor,
        name="resnet",
        trainable=True,
        dtype=tf.float32)

    model.build((None, None, None, 3))
    logger.info(f'Loading weights...')
    model.load_weights(bit_model_file)
    logger.info(f'Weights loaded into model!')

    model._head = tf.keras.layers.Dense(
        units=dataset_info['num_classes'],
        use_bias=True,
        kernel_initializer="zeros",
        trainable=True,
        name="head/dense")

    lr_supports = bit_hyperrule.get_schedule(dataset_info['num_examples'])

    schedule_length = lr_supports[-1]
    # NOTE: Let's not do that unless verified necessary and we do the same
    # across all three codebases.
    # schedule_length = schedule_length * 512 / args.batch

    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

  logger.info(f'Fine-tuning the model...')
  steps_per_epoch = args.eval_every or schedule_length
  history = model.fit(
      data_train,
      steps_per_epoch=steps_per_epoch,
      epochs=schedule_length // steps_per_epoch,
      validation_data=data_test,  # here we are only using
                                  # this data to evaluate our performance
      callbacks=[BiTLRSched(args.base_lr, dataset_info['num_examples'])],
  )

  for epoch, accu in enumerate(history.history['val_accuracy']):
    logger.info(
            f'Step: {epoch * args.eval_every}, '
            f'Test accuracy: {accu:0.3f}')


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--tfds_manual_dir", default=None,
                      help="Path to maually downloaded dataset.")
  parser.add_argument("--batch_eval", default=32, type=int,
                      help="Eval batch size.")
  main(parser.parse_args())
