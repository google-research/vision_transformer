# Copyright 2026 Google LLC.
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

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp

from vit_jax import checkpoint
from vit_jax import models
from vit_jax import test_utils
from vit_jax.configs import models as config_lib


class InspectParamsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('top_level', {'empty': {}}),
      ('nested', {'block': {'empty': {}}}),
      ('deep', {'block': {'layer': {'empty': {}}}}),
      ('siblings', {'block': {'first': {}, 'second': {}}}),
  )
  def test_restores_empty_subtrees(self, empty_subtrees):
    kernel = jnp.array([1., 2.])
    params = {'block': {'kernel': kernel}}
    expected = {'block': {'kernel': kernel}}
    for key, value in empty_subtrees.items():
      if key == 'block':
        expected['block'].update(value)
      else:
        expected[key] = value
    result = checkpoint.inspect_params(params=params, expected=expected)
    self.assertEqual(jax.tree.structure(result), jax.tree.structure(expected))
    self.assertIs(result['block']['kernel'], kernel)
    # The reconstructed tree must be usable as a Flax serialization state.
    restored = flax.serialization.from_state_dict(expected, result)
    self.assertEqual(jax.tree.structure(restored), jax.tree.structure(expected))

  def test_rejects_missing_nonempty_parameters(self):
    with self.assertRaisesRegex(ValueError, 'Missing params from checkpoint'):
      checkpoint.inspect_params(
          params={}, expected={'block': {'kernel': jnp.ones(2), 'empty': {}}})

  def test_rejects_extra_parameters(self):
    with self.assertRaisesRegex(ValueError, 'Extra params in checkpoint'):
      checkpoint.inspect_params(
          params={'unexpected': jnp.ones(2)}, expected={'block': {'empty': {}}})

  def test_permissive_inspection_preserves_existing_parameters(self):
    kernel = jnp.ones(2)
    result = checkpoint.inspect_params(
        params={'block': {'kernel': kernel}},
        expected={'block': {'empty': {}}, 'missing': jnp.zeros(2)},
        fail_if_extra=False, fail_if_missing=False)
    self.assertEqual(set(result), {'block'})
    self.assertEqual(set(result['block']), {'kernel', 'empty'})
    self.assertIs(result['block']['kernel'], kernel)


class CheckpointTest(absltest.TestCase):

  def test_load_pretrained(self):
    tempdir = tempfile.gettempdir()
    model_config = config_lib.get_testing_config()
    test_utils.create_checkpoint(model_config, f'{tempdir}/testing.npz')
    model = models.VisionTransformer(num_classes=2, **model_config)
    variables = model.init(
        jax.random.PRNGKey(0),
        inputs=jnp.ones([1, 32, 32, 3], jnp.float32),
        train=False,
    )
    checkpoint.load_pretrained(
        pretrained_path=f'{tempdir}/testing.npz',
        init_params=variables['params'],
        model_config=model_config)


if __name__ == '__main__':
  absltest.main()
