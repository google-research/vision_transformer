## Big Transfer (BiT): General Visual Representation Learning
*by Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby*

## Introduction

In this repository we release multiple models from the [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370) paper that were pre-trained on the [ILSVRC-2012](http://www.image-net.org/challenges/LSVRC/2012/) and [ImageNet-21k](http://www.image-net.org/) datasets.
We provide the code to fine-tuning the released models in the major deep learning frameworks [TensorFlow 2](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/) and [Jax](https://jax.readthedocs.io/en/latest/index.html)/[Flax](http://flax.readthedocs.io).

We hope that the computer vision community will benefit by employing more powerful ImageNet-21k pretrained models as opposed to conventional models pre-trained on the ILSVRC-2012 dataset.

We also provide colabs for a more exploratory interactive use:
a [TensorFlow 2 colab](https://colab.research.google.com/github/google-research/big_transfer/blob/master/colabs/big_transfer_tf2.ipynb),
a [PyTorch colab](https://colab.research.google.com/github/google-research/big_transfer/blob/master/colabs/big_transfer_pytorch.ipynb),
and a [Jax colab](https://colab.research.google.com/github/google-research/big_transfer/blob/master/colabs/big_transfer_jax.ipynb).

## Installation

Make sure you have `Python>=3.6` installed on your machine.

To setup [Tensorflow 2](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch) or [Jax](https://github.com/google/jax), follow the instructions provided in the corresponding repository linked here.

In addition, install python dependencies by running (please select `tf2`, `pytorch` or `jax` in the command below):
```
pip install -r bit_{tf2|pytorch|jax}/requirements.txt
```

## How to fine-tune BiT
First, download the BiT model. We provide models pre-trained on ILSVRC-2012 (BiT-S) or ImageNet-21k (BiT-M) for 5 different architectures: ResNet-50x1, ResNet-101x1, ResNet-50x3, ResNet-101x3, and ResNet-152x4.

For example, if you would like to download the ResNet-50x1 pre-trained on ImageNet-21k, run the following command:
```
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.{npz|h5}
```
Other models can be downloaded accordingly by plugging the name of the model (BiT-S or BiT-M) and architecture in the above command.
Note that we provide models in two formats: `npz` (for PyTorch and Jax) and `h5` (for TF2). By default we expect that model weights are stored in the root folder of this repository.

Then, you can run fine-tuning of the downloaded model on your dataset of interest in any of the three frameworks. All frameworks share the command line interface
```
python3 -m bit_{pytorch|jax|tf2}.train --name cifar10_`date +%F_%H%M%S` --model BiT-M-R50x1 --logdir /tmp/bit_logs --dataset cifar10
```
Currently. all frameworks will automatically download CIFAR-10 and CIFAR-100 datasets. Other public or custom datasets can be easily integrated: in TF2 and JAX we rely on the extensible [tensorflow datasets library](https://github.com/tensorflow/datasets/). In PyTorch, we use [torchvision’s data input pipeline](https://pytorch.org/docs/stable/torchvision/index.html).

Note that our code uses all available GPUs for fine-tuning.

We also support training in the low-data regime: the `--examples_per_class <K>` option will randomly draw K samples per class for training.

To see a detailed list of all available flags, run `python3 -m bit_{pytorch|jax|tf2}.train --help`.

### BiT-M models fine-tuned on ILSVRC-2012

For convenience, we provide BiT-M models that were already fine-tuned on the
ILSVRC-2012 dataset. The models can be downloaded by adding the `-ILSVRC2012`
postfix, e.g.
```
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1-ILSVRC2012.npz
```

### Available architectures

We release all architectures mentioned in the paper, such that you may choose between accuracy or speed: R50x1, R101x1, R50x3, R101x3, R152x4.
In the above path to the model file, simply replace `R50x1` by your architecture of choice.

We further investigated more architectures after the paper's publication and found R152x2 to have a nice trade-off between speed and accuracy, hence we also include this in the release and provide a few numbers below.


### Hyper-parameters

For reproducibility, our training script uses hyper-parameters (BiT-HyperRule) that were used in the original paper.
Note, however, that BiT models were trained and finetuned using Cloud TPU hardware, so for a typical GPU setup our default hyper-parameters could require too much memory or result in a very slow progress.
Moreover, BiT-HyperRule is designed to generalize across many datasets, so it is typically possible to devise more efficient application-specific hyper-parameters.
Thus, we encourage the user to try more light-weight settings, as they require much less resources and often result in a similar accuracy.

For example, we tested our code using a 8xV100 GPU machine on the CIFAR-10 and CIFAR-100 datasets, while reducing batch size from 512 to 128 and learning rate from 0.003 to 0.001.
This setup resulted in nearly identical performance (see [Expected results](#expected-results) below) in comparison to BiT-HyperRule, despite being less computationally demanding.

Below, we provide more suggestions on how to optimize our paper's setup.

### Tips for optimizing memory or speed

The default BiT-HyperRule was developed on Cloud TPUs and is quite memory-hungry.
This is mainly due to the large batch-size (512) and image resolution (up to 480x480).
Here are some tips if you are running out of memory:

  1. In `bit_hyperrule.py` we specify the input resolution.
     By reducing it, one can save a lot of memory and compute, at the expense of accuracy.
  2. The batch-size can be reduced in order to reduce memory consumption.
     However, one then also needs to play with learning-rate and schedule (steps) in order to maintain the desired accuracy.
  3. The PyTorch codebase supports a batch-splitting technique ("micro-batching") via `--batch_split` option.
     For example, running the fine-tuning with `--batch_split 8` reduces memory requirement by a factor of 8.

## Expected results

We verified that when using the BiT-HyperRule, the code in this repository reproduces the paper's results.

### CIFAR results (few-shot and full)

For these common benchmarks, the aforementioned changes to the BiT-HyperRule (`--batch 128 --base_lr 0.001`) lead to the following, very similar results.
The table shows the min←**median**→max result of at least five runs.
**NOTE**: This is not a comparison of frameworks, just evidence that all code-bases can be trusted to reproduce results.

#### BiT-M-R101x3

| Dataset  | Ex/cls |          TF2           |          Jax           |         PyTorch        |
| :---     | :---:  |         :---:          |         :---:          |          :---:         |
| CIFAR10  |   1    | 52.5 ← **55.8** → 60.2 | 48.7 ← **53.9** → 65.0 | 56.4 ← **56.7** → 73.1 |
| CIFAR10  |   5    | 85.3 ← **87.2** → 89.1 | 80.2 ← **85.8** → 88.6 | 84.8 ← **85.8** → 89.6 |
| CIFAR10  |  full  |        **98.5**        |        **98.4**        | 98.5 ← **98.6** → 98.6 |
| CIFAR100 |   1    | 34.8 ← **35.7** → 37.9 | 32.1 ← **35.0** → 37.1 | 31.6 ← **33.8** → 36.9 |
| CIFAR100 |   5    | 68.8 ← **70.4** → 71.4 | 68.6 ← **70.8** → 71.6 | 70.6 ← **71.6** → 71.7 |
| CIFAR100 |  full  |        **90.8**        |        **91.2**        | 91.1 ← **91.2** → 91.4 |

#### BiT-M-R152x2

| Dataset  | Ex/cls |           Jax          |         PyTorch        |
| :---     | :---:  |          :---:         |          :---:         |
| CIFAR10  |   1    | 44.0 ← **56.7** → 65.0 | 50.9 ← **55.5** → 59.5 |
| CIFAR10  |   5    | 85.3 ← **87.0** → 88.2 | 85.3 ← **85.8** → 88.6 |
| CIFAR10  |  full  |        **98.5**        | 98.5 ← **98.5** → 98.6 |
| CIFAR100 |   1    | 36.4 ← **37.2** → 38.9 | 34.3 ← **36.8** → 39.0 |
| CIFAR100 |   5    | 69.3 ← **70.5** → 72.0 | 70.3 ← **72.0** → 72.3 |
| CIFAR100 |  full  |        **91.2**        | 91.2 ← **91.3** → 91.4 |

(TF2 models not yet available.)

#### BiT-M-R50x1

| Dataset  | Ex/cls |          TF2           |          Jax           |         PyTorch        |
| :---     | :---:  |         :---:          |         :---:          |          :---:         |
| CIFAR10  |   1    | 49.9 ← **54.4** → 60.2 | 48.4 ← **54.1** → 66.1 | 45.8 ← **57.9** → 65.7 |
| CIFAR10  |   5    | 80.8 ← **83.3** → 85.5 | 76.7 ← **82.4** → 85.4 | 80.3 ← **82.3** → 84.9 |
| CIFAR10  |  full  |        **97.2**        |        **97.3**        |        **97.4**        |
| CIFAR100 |   1    | 35.3 ← **37.1** → 38.2 | 32.0 ← **35.2** → 37.8 | 34.6 ← **35.2** → 38.6 |
| CIFAR100 |   5    | 63.8 ← **65.0** → 66.5 | 63.4 ← **64.8** → 66.5 | 64.7 ← **65.5** → 66.0 |
| CIFAR100 |  full  |        **86.5**        |        **86.4**        |        **86.6**        |

### ImageNet results

These results were obtained using BiT-HyperRule.
However, because this results in large batch-size and large resolution, memory can be an issue.
The PyTorch code supports batch-splitting, and hence we can still run things there without resorting to Cloud TPUs by adding the `--batch_split N` command where `N` is a power of two.
For instance, the following command produces a validation accuracy of `80.68` on a machine with 8 V100 GPUs:

```
python3 -m bit_pytorch.train --name ilsvrc_`date +%F_%H%M%S` --model BiT-M-R50x1 --logdir /tmp/bit_logs --dataset imagenet2012 --batch_split 4
```

Further increase to `--batch_split 8` when running with 4 V100 GPUs, etc.

Full results achieved that way in some test runs were:

| Ex/cls | R50x1 | R152x2 | R101x3 |
| :---:  | :---: | :---:  | :---:  |
|   1    | 18.36 | 24.5   | 25.55  |
|   5    | 50.64 | 64.5   | 64.18  |
|  full  | 80.68 | WIP    | WIP    |
