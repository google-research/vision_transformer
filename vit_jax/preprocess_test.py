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

import contextlib
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from vit_jax import preprocess

VOCAB = """[PAD]
[CLS]
some
test
words"""


@contextlib.contextmanager
def _create_vocab():
  with tempfile.NamedTemporaryFile('w') as f:
    f.write(VOCAB)
    f.flush()
    yield f.name


class PreprocessTest(absltest.TestCase):

  def test_bert_tokenizer(self):
    with _create_vocab() as vocab_path:
      tokenizer = preprocess.BertTokenizer(vocab_path=vocab_path, max_len=3)

    tokens = tokenizer(['some', 'test', 'words', 'xxx'])
    np.testing.assert_equal(tokens, [
        [1, 2, 0],
        [1, 3, 0],
        [1, 4, 0],
        [1, 5, 0],
    ])

  @mock.patch('tensorflow_text.SentencepieceTokenizer')
  @mock.patch('tensorflow.io.gfile.GFile')
  def test_sentencepiece_tokenizer(self, gfile_patch, tokenizer_patch):
    gfile_patch.return_value.read.return_value = 'test vocab'
    eos_token = 7
    tokenizer_patch.return_value.string_to_id.return_value = eos_token
    tokenizer_patch.return_value.tokenize.side_effect = (
        tf.constant([1, eos_token], tf.int32),
        tf.constant([2, 3, eos_token], tf.int32),
        tf.constant([4, 5, 6, eos_token], tf.int32),
    )

    tokenizer = preprocess.SentencepieceTokenizer(
        vocab_path='test_path', max_len=3)
    gfile_patch.assert_called_once_with('test_path', 'rb')
    tokenizer_patch.assert_called_once_with(model='test vocab', add_eos=True)
    tokenizer_patch.return_value.string_to_id.assert_called_once_with('</s>')

    tokens = tokenizer(['some', 'test', 'words'])
    tokenizer_patch.return_value.tokenize.assert_has_calls(
        (mock.call('some'), mock.call('test'), mock.call('words')))
    np.testing.assert_equal(tokens, [
        [1, eos_token, eos_token],
        [2, 3, eos_token],
        [4, 5, eos_token],
    ])

  def test_preprocess_images(self):
    # white images with black border
    img1 = 255 * np.concatenate([  # portrait image
        np.zeros([2, 10, 3], np.uint8),
        np.ones([12, 10, 3], np.uint8),
        np.zeros([2, 10, 3], np.uint8),
    ], axis=0)
    img2 = 255 * np.concatenate([  # landscape image
        np.zeros([10, 2, 3], np.uint8),
        np.ones([10, 12, 3], np.uint8),
        np.zeros([10, 2, 3], np.uint8),
    ], axis=1)

    preprocess_images = preprocess.PreprocessImages(size=4, crop=False)
    imgs = preprocess_images([img1, img2])
    self.assertEqual(imgs.shape, (2, 4, 4, 3))
    self.assertLess(imgs.mean(), 1.0)  # borders resized

    preprocess_images = preprocess.PreprocessImages(size=4, crop=True)
    imgs = preprocess_images([img1, img2])
    self.assertEqual(imgs.shape, (2, 4, 4, 3))
    self.assertEqual(imgs.mean(), 1.0)  # borders cropped

  def test_pp_bert(self):
    with _create_vocab() as vocab_path:
      pp = preprocess.get_pp(
          tokenizer_name='bert', vocab_path=vocab_path, max_len=3, size=4)

    ds = tf.data.Dataset.from_tensor_slices({
        'text':
            tf.constant(['test', 'test']),
        'image': [
            tf.ones([10, 10, 3], tf.uint8),
            tf.ones([10, 10, 3], tf.uint8)
        ],
    })

    b = next(iter(ds.map(pp).batch(2).as_numpy_iterator()))
    dtypes_shapes = {k: (v.dtype, v.shape) for k, v in b.items()}
    np.testing.assert_equal(dtypes_shapes, {
        'image': (np.float32, (2, 4, 4, 3)),
        'text': (object, (2,)),
        'tokens': (np.int32, (2, 3))
    })

  @mock.patch('tensorflow_text.SentencepieceTokenizer')
  @mock.patch('tensorflow.io.gfile.GFile')
  def test_pp_sentencepiece(self, gfile_patch, tokenizer_patch):
    eos_token = 7
    gfile_patch.return_value.read.return_value = 'test vocab'
    tokenizer_patch.return_value.string_to_id.return_value = eos_token
    tokenizer_patch.return_value.tokenize.side_effect = (
        tf.constant([1, eos_token], tf.int32),
        tf.constant([2, 3, eos_token], tf.int32),
    )
    pp = preprocess.get_pp(
        tokenizer_name='sentencepiece',
        vocab_path='test',
        max_len=3,
        size=4)

    ds = tf.data.Dataset.from_tensor_slices({
        'text':
            tf.constant(['test', 'test']),
        'image': [
            tf.ones([10, 10, 3], tf.uint8),
            tf.ones([10, 10, 3], tf.uint8)
        ],
    })

    b = next(iter(ds.map(pp).batch(2).as_numpy_iterator()))
    dtypes_shapes = {k: (v.dtype, v.shape) for k, v in b.items()}
    np.testing.assert_equal(dtypes_shapes, {
        'image': (np.float32, (2, 4, 4, 3)),
        'text': (object, (2,)),
        'tokens': (np.int32, (2, 3))
    })

if __name__ == '__main__':
  absltest.main()
