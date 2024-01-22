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

"""Preprocessing utilities for text/image models."""

import dataclasses

import numpy as np
import tensorflow as tf
import tensorflow_text

def get_tokenizer(tokenizer_name):
  """Returns a tokenizer specified by name ("bert" or "sentencpiece")."""
  return {
      'bert': BertTokenizer,
      'sentencepiece': SentencepieceTokenizer,
  }[tokenizer_name]


@dataclasses.dataclass(frozen=True)
class BertTokenizer:
  """BERT tokenizer with prepended CLS token and fixed sequence length.

  This class can be used to tokenize batches of text tokens to numpy arrays
  (by calling `__call__()`), or as part of a TensorFlow preprocessing graph
  (via the method `preprocess_tf()`).

  Attributes:
    vocab_path: Path pointing to the vocabulary file. Can be any path string
      that is understood by `tf.io.gfile`.
    max_len: Length of tokenized sequences. If the provided texts result in
      fewer tokens, then the sequence is zero-padded. If the provided texts
      result in more tokens, then the tokens are clipped.
    cls_token: Will be set during class construction.
  """

  vocab_path: str
  max_len: int
  cls_token: int = dataclasses.field(init=False)

  _tokenizer: tensorflow_text.BertTokenizer = dataclasses.field(init=False)

  def __post_init__(self):
    tokenizer = tensorflow_text.BertTokenizer(
        self.vocab_path, token_out_type=tf.int32, lower_case=True)
    with tf.io.gfile.GFile(self.vocab_path) as f:
      vocab = f.read().split('\n')
    cls_token = vocab.index('[CLS]')

    # Work-around for frozen dataclasses:
    # https://stackoverflow.com/questions/53756788
    object.__setattr__(self, 'cls_token', cls_token)
    object.__setattr__(self, '_tokenizer', tokenizer)

  def preprocess_tf(self, text):
    """Tokenizes a single text as part of a TensorFlow graph."""
    return self._preprocess(text[None])[0]

  def _preprocess(self, texts):
    token_ids = self._tokenizer.tokenize(texts)
    tokens, mask = tensorflow_text.pad_model_inputs(token_ids, self.max_len - 1)
    del mask  # Recovered from zero padding in model.
    count = tf.shape(tokens)[0]
    return tf.concat([tf.fill([count, 1], self.cls_token), tokens], axis=1)

  def __call__(self, texts):
    """Tokenizes a batch of texts to a numpy array."""
    return self._preprocess(tf.constant(texts)).numpy()


@dataclasses.dataclass(frozen=True)
class SentencepieceTokenizer:
  """SentencePiece tokenizer with sticky eos.

  Models that use this tokanizer usually use the *last* token, which is
  guaranteed to be the "</s>" token (even if tokens are capped to `max_len`).
  The same token is used for padding (and exposed as `eos_token`).

  This class can be used to tokenize batches of text tokens to numpy arrays
  (by calling `__call__()`), or as part of a TensorFlow preprocessing graph
  (via the method `preprocess_tf()`).

  Attributes:
    vocab_path: Path pointing to the vocabulary file. Can be any path string
      that is understood by `tf.io.gfile`.
    max_len: Length of tokenized sequences. If the provided texts result in
      fewer tokens, then the sequence is zero-padded. If the provided texts
      result in more tokens, then the tokens are clipped.
    eos_token: Token used for padding. Last token is guaranteed to be padded.
  """

  vocab_path: str
  max_len: int
  eos_token: int = dataclasses.field(init=False)

  _tokenizer: tensorflow_text.BertTokenizer = dataclasses.field(init=False)

  def __post_init__(self):
    tokenizer = tensorflow_text.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(self.vocab_path, 'rb').read(), add_eos=True)
    eos_token = tokenizer.string_to_id('</s>')

    # Work-around for frozen dataclasses:
    # https://stackoverflow.com/questions/53756788
    object.__setattr__(self, 'eos_token', eos_token)
    object.__setattr__(self, '_tokenizer', tokenizer)

  def preprocess_tf(self, text):
    """Tokenizes a single text as part of a TensorFlow graph."""
    tokens = self._tokenizer.tokenize(text)
    tokens = tokens[:self.max_len - 1]  # to guarantee eos at end
    return tf.pad(
        tokens, [(0, self.max_len - tf.shape(tokens)[0])],
        constant_values=self.eos_token)

  def __call__(self, texts):
    """Tokenizes a batch of texts to a numpy array."""
    return tf.stack([self.preprocess_tf(text) for text in texts]).numpy()


@dataclasses.dataclass(frozen=True)
class PreprocessImages:
  """Resizes images and sets value range to [-1, 1].

  This class can be used to tokenize batches of text tokens to numpy arrays
  (by calling `__call__()`), or as part of a TensorFlow preprocessing graph
  (via the method `preprocess_tf()`).

  Attributes:
    size: Target size of images.
    crop: If set to true, then the image will first be resized maintaining the
      original aspect ratio, and then a central crop of that resized image will
      be returned.
  """
  size: int
  crop: bool = False

  def _resize_small(self, image):  # pylint: disable=missing-docstring
    h, w = tf.shape(image)[0], tf.shape(image)[1]

    # Figure out the necessary h/w.
    ratio = (
        tf.cast(self.size, tf.float32) /
        tf.cast(tf.minimum(h, w), tf.float32))
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

    return tf.image.resize(image, (h, w), method='bilinear')

  def _crop(self, image):
    h, w = self.size, self.size
    dy = (tf.shape(image)[0] - h) // 2
    dx = (tf.shape(image)[1] - w) // 2
    return tf.image.crop_to_bounding_box(image, dy, dx, h, w)

  def _resize(self, image):
    return tf.image.resize(
        image, size=[self.size, self.size], method='bilinear')

  def _value_range(self, image):
    image = tf.cast(image, tf.float32) / 255
    return -1 + image * 2

  def preprocess_tf(self, image):
    """Resizes a single image as part of a TensorFlowg graph."""
    assert image.dtype == tf.uint8
    if self.crop:
      image = self._resize_small(image)
      image = self._crop(image)
    else:
      image = self._resize(image)
    image = tf.cast(image, tf.uint8)
    return self._value_range(image)

  def __call__(self, images):
    """Resizes a sequence of images, returns a numpy array."""
    return np.stack([
        self.preprocess_tf(tf.constant(image)) for image in images
    ])


def get_pp(*, tokenizer_name, vocab_path, max_len, size, crop=False):
  """Returns preprocessing function for "image" and "text" features.

  The returned function can directly be used with `tf.data.Dataset.map()`.
  If either the text feature (feature key "text") or the image feature (feature
  key "image") are not found, then they will be left untouched.

  Note that the "image" feature is overwritten with the resized image, but the
  "text" feature is tokenized into a new feature "tokens".

  Args:
    tokenizer_name: Name of tokenizer (either "bert", or "sentencepiece").
    vocab_path: Argument passed to tokenizer.
    max_len: Argument passed to tokenizer.
    size: Argument passed to `PreprocessImages`.
    crop: Argument passed to `PreprocessImages`.
  """
  tokenizer_class = get_tokenizer(tokenizer_name)
  tokenizer = tokenizer_class(vocab_path=vocab_path, max_len=max_len)
  preprocess_images = PreprocessImages(size=size, crop=crop)

  def pp(features):
    features = {**features}
    if 'image' in features:
      features['image'] = preprocess_images.preprocess_tf(features['image'])
    if 'text' in features:
      features['tokens'] = tokenizer.preprocess_tf(features['text'])
    return features

  return pp
