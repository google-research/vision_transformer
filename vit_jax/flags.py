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

import argparse


def argparser(known_models, known_datasets):
  parser = argparse.ArgumentParser(description='Fine-tune ViT-M model.')
  parser.add_argument(
      '--name',
      required=True,
      help='Name of this run. Used for monitoring and checkpointing.')
  parser.add_argument(
      '--model',
      choices=list(known_models),
      help='Which variant to use; ViT-M gives best results.')
  parser.add_argument(
      '--logdir', required=True, help='Where to log training info (small).')
  parser.add_argument(
      '--vit_pretrained_dir',
      default='.',
      help='Where to search for pretrained ViT models.')
  parser.add_argument(
      '--output',
      default=None,
      help='Where to store the fine tuned model checkpoint.')
  parser.add_argument(
      '--copy_to',
      default=None,
      help='Directory where --logdir and --output should be stored. This '
      'directory can be on any filesystem accessible through by tf.io.gfile',
  )

  parser.add_argument(
      '--dataset',
      choices=list(known_datasets),
      required=True,
      help='Choose the dataset. It should be easy to add your own! '
      'Do not forget to set --tfds_manual_dir if necessary.')
  parser.add_argument(
      '--tfds_manual_dir',
      default=None,
      help='Path to manually downloaded dataset.')

  parser.add_argument(
      '--mixup_alpha',
      type=float,
      default=0,
      help='Coefficient for mixup combination. See https://arxiv.org/abs/1710.09412'
  )
  parser.add_argument(
      '--grad_norm_clip', type=int, default=1, help='Resizes global gradients.')

  parser.add_argument(
      '--total_steps',
      type=int,
      default=None,
      help='Number of steps; determined by hyper module if not specified.')
  parser.add_argument(
      '--accum_steps',
      type=int,
      default=8,
      help='Accumulate gradients over multiple steps to save on memory.')
  parser.add_argument(
      '--batch', type=int, default=512, help='Batch size for training.')
  parser.add_argument(
      '--batch_eval', type=int, default=512, help='Batch size for evaluation.')
  parser.add_argument(
      '--shuffle_buffer',
      type=int,
      default=200_000,
      help='Shuffle buffer size.')
  parser.add_argument(
      '--prefetch',
      type=int,
      default=2,
      help='Number of batches to prefetch to device.')

  parser.add_argument(
      '--base_lr',
      type=float,
      default=0.03,
      help='Base learning-rate for fine-tuning. Most likely default is best.')
  parser.add_argument(
      '--decay_type',
      choices=['cosine', 'linear'],
      default='cosine',
      help='How to decay the learning rate.')
  parser.add_argument(
      '--warmup_steps',
      type=int,
      default=500,
      help='How to decay the learning rate.')

  parser.add_argument(
      '--eval_every',
      type=int,
      default=100,
      help='Run prediction on validation set every so many steps.'
      'Will always run one evaluation at the end of training.')
  parser.add_argument(
      '--progress_every',
      type=int,
      default=10,
      help='Log progress every so many steps.')

  return parser
