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

"""setup.py for vision_transformer repo, vit_jax package."""

import os
from setuptools import find_packages
from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, 'README.md'), encoding='utf-8').read()
except IOError:
  README = ''

install_requires = [
    'absl-py',
    'aqtp!=0.1.1',  # https://github.com/google/aqt/issues/196
    'clu',
    'einops',
    'flax',
    'flaxformer @ git+https://github.com/google/flaxformer',
    'jax',
    'ml-collections',
    'numpy',
    'packaging',
    'pandas',
    'scipy',
    'tensorflow_datasets',
    'tensorflow_probability',
    'tensorflow',
    'tensorflow_text',
    'tqdm',
]

tests_require = [
    'pytest',
]

__version__ = None

with open(os.path.join(here, 'version.py')) as f:
  exec(f.read(), globals())  # pylint: disable=exec-used

setup(
    name='vit_jax',
    version=__version__,
    description='Original JAX implementation of Vision Transformer models.',
    long_description=README,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    keywords='',
    author='Vision Transformer Authors',
    author_email='no-reply@google.com',
    url='https://github.com/google-research/vision_transformer',
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=dict(test=tests_require),
    )
