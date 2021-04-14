# Vision Transformer
by Alexey Dosovitskiy\*†, Lucas Beyer\*, Alexander Kolesnikov\*, Dirk
Weissenborn\*, Xiaohua Zhai\*, Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby\*†.

(\*) equal technical contribution, (†) equal advising.

Open source release prepared by Andreas Steiner.

Note: This repository was forked and modified from
[google-research/big_transfer](https://github.com/google-research/big_transfer).

## Introduction

In this repository we release models from the paper [An Image is Worth 16x16
Words: Transformers for Image Recognition at
Scale](https://arxiv.org/abs/2010.11929) that were pre-trained on the
[ImageNet-21k](http://www.image-net.org/) (`imagenet21k`) dataset. We provide
the code for fine-tuning the released models in
[Jax](https://jax.readthedocs.io)/[Flax](http://flax.readthedocs.io).

![Figure 1 from paper](figure1.png)

Overview of the model: we split an image into fixed-size patches, linearly embed
each of them, add position embeddings, and feed the resulting sequence of
vectors to a standard Transformer encoder. In order to perform classification,
we use the standard approach of adding an extra learnable "classification token"
to the sequence.

## Colab

Check out the Colab for loading the data, fine-tuning the model, evaluation,
and inference. The Colab loads the code from this repository and runs by
default on a TPU with 8 cores.

https://colab.research.google.com/github/google-research/vision_transformer/blob/master/vit_jax.ipynb

Note that the Colab can be run as is storing all data in the ephemeral VM, or,
alternatively you can log into your personal Google Drive to persist the code
and data there.

## Installation

Make sure you have `Python>=3.6` installed on your machine.

For installing [Jax](https://github.com/google/jax), follow the instructions
provided in the corresponding repository linked here. Note that installation
instructions for GPU differs slightly from the instructions for CPU.

Then, install python dependencies by running:
```
pip install -r vit_jax/requirements.txt
```

## Available models

We provide models pre-trained on imagenet21k for the following architectures:
ViT-B/16, ViT-B/32, ViT-L/16 and ViT-L/32. We  provide the same models
pre-trained on imagenet21k *and* fine-tuned on imagenet2012.

**Update (1.12.2020)**: We have added the R50+ViT-B/16 hybrid model (ViT-B/16 on
top of a Resnet-50 backbone). When pretrained on imagenet21k, this model
achieves almost the performance of the L/16 model with less than half the
computational finetuning cost. Note that "R50" is somewhat modified for the B/16
variant: The original ResNet-50 has [3,4,6,3] blocks, each reducing the
resolution of the image by a factor of two. In combination with the ResNet stem
this would result in a reduction of 32x so even with a patch size of (1,1) the
ViT-B/16 variant cannot be realized anymore. For this reason we instead use
[3,4,9] blocks for the R50+B/16 variant.

**Update (9.11.2020)**: We have also added the ViT-L/16 model.

**Update (29.10.2020)**: We have added ViT-B/16 and ViT-L/16 models pretrained
on ImageNet-21k and then fine-tuned on ImageNet at 224x224 resolution (instead
of default 384x384). These models have the suffix "-224" in their name.
They are expected to achieve 81.2% and 82.7% top-1 accuracies respectively.

You can find all these models in the following storage bucket:

https://console.cloud.google.com/storage/vit_models/

For example, if you would like to download the ViT-B/16 pre-trained on
imagenet21k run the following command:

```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

## How to fine-tune ViT

You can run fine-tuning of the downloaded model on your dataset of interest. All
frameworks share the command line interface

```
python3 -m vit_jax.train --name ViT-B_16-cifar10_`date +%F_%H%M%S` --model ViT-B_16 --logdir /tmp/vit_logs --dataset cifar10
```

Currently, the code will automatically download CIFAR-10 and CIFAR-100 datasets.
Other public or custom datasets can be easily integrated, using [tensorflow
datasets library](https://github.com/tensorflow/datasets/). Note that you will
also need to update `vit_jax/input_pipeline.py` to specify some parameters about
any added dataset.

Note that our code uses all available GPUs/TPUs for fine-tuning.

To see a detailed list of all available flags, run `python3 -m vit_jax.train
--help`.

Notes about some flags:

  - `--accum_steps=16` : This works well with ViT-B_16 on a machine that has 8
    GPUs of type V100 with 16G memory each attached. If you have fewer
    accelerators or accelerators with less memory, you can use the same
    configuration but increase the `--accum_steps`. For a small model like
    ViT-B_32 you can even use `--accum_steps=1`. For a large model like ViT-L_16
    you need to go in the other direction (e.g. `--accum_steps=32`). Note that
    the largest model ViT-H_14 also needs adaptation of the batch size
    (`--accum_steps=2 --batch=16` should work on a 8x V100).
    tested `)
  - `--batch=512` : Alternatively, you can decrease the batch size, but
    that usually involves some tuning of the learning rate parameters.

## Expected results

In this table we closely follow experiments from the paper and report results
that were achieved by running this code on Google Cloud machine with eight V100
GPUs.

Note: Runs in table below before 2020-11-03 ([6fba202]) have
`config.transformer.dropout_rate=0.0`.

[6fba202]: https://github.com/google-research/vision_transformer/commit/6fba202f04622a17c6361a8c81ef471540facaa7

| upstream    | model        | dataset      |   accuracy | wall_clock_time   | link                                                                                                                                                            |
|:------------|:-------------|:-------------|-----------:|:------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| imagenet21k | R50+ViT-B_16 | cifar10      |     0.9893 | 10.8h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/R50%5C%2BViT-B_16/cifar10/)      |
| imagenet21k | R50+ViT-B_16 | cifar10      |     0.9885 | 10.9h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/R50%5C%2BViT-B_16/cifar10/)      |
| imagenet21k | R50+ViT-B_16 | cifar100     |     0.9235 | 10.8h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/R50%5C%2BViT-B_16/cifar100/)     |
| imagenet21k | R50+ViT-B_16 | cifar100     |     0.9239 | 10.8h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/R50%5C%2BViT-B_16/cifar100/)     |
| imagenet21k | R50+ViT-B_16 | imagenet2012 |     0.8505 | 25.9h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/R50%5C%2BViT-B_16/imagenet2012/) |
| imagenet21k | R50+ViT-B_16 | imagenet2012 |     0.8492 | 25.9h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/R50%5C%2BViT-B_16/imagenet2012/) |
| imagenet21k | ViT-B_16     | cifar10      |     0.9892 | 7.2h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_16/cifar10/)               |
| imagenet21k | ViT-B_16     | cifar10      |     0.9903 | 7.7h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_16/cifar10/)               |
| imagenet21k | ViT-B_16     | cifar100     |     0.9226 | 7.2h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_16/cifar100/)              |
| imagenet21k | ViT-B_16     | cifar100     |     0.9264 | 7.5h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_16/cifar100/)              |
| imagenet21k | ViT-B_16     | imagenet2012 |     0.8462 | 17.9h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_16/imagenet2012/)          |
| imagenet21k | ViT-B_16     | imagenet2012 |     0.8461 | 17.8h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_16/imagenet2012/)          |
| imagenet21k | ViT-B_32     | cifar10      |     0.9893 | 1.6h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_32/cifar10/)               |
| imagenet21k | ViT-B_32     | cifar10      |     0.9889 | 1.6h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_32/cifar10/)               |
| imagenet21k | ViT-B_32     | cifar100     |     0.9208 | 1.6h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_32/cifar100/)              |
| imagenet21k | ViT-B_32     | cifar100     |     0.9196 | 1.6h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_32/cifar100/)              |
| imagenet21k | ViT-B_32     | imagenet2012 |     0.8179 | 4.2h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_32/imagenet2012/)          |
| imagenet21k | ViT-B_32     | imagenet2012 |     0.8179 | 4.1h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-B_32/imagenet2012/)          |
| imagenet21k | ViT-L_16     | cifar10      |     0.9907 | 24.7h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_16/cifar10/)               |
| imagenet21k | ViT-L_16     | cifar10      |     0.991  | 24.9h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_16/cifar10/)               |
| imagenet21k | ViT-L_16     | cifar100     |     0.9304 | 24.8h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_16/cifar100/)              |
| imagenet21k | ViT-L_16     | cifar100     |     0.93   | 24.4h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_16/cifar100/)              |
| imagenet21k | ViT-L_16     | imagenet2012 |     0.8507 | 59.2h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_16/imagenet2012/)          |
| imagenet21k | ViT-L_16     | imagenet2012 |     0.8505 | 59.2h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_16/imagenet2012/)          |
| imagenet21k | ViT-L_32     | cifar10      |     0.9903 | 5.7h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_32/cifar10/)               |
| imagenet21k | ViT-L_32     | cifar10      |     0.9909 | 5.8h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_32/cifar10/)               |
| imagenet21k | ViT-L_32     | cifar100     |     0.9302 | 6.7h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_32/cifar100/)              |
| imagenet21k | ViT-L_32     | cifar100     |     0.9306 | 6.7h              | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_32/cifar100/)              |
| imagenet21k | ViT-L_32     | imagenet2012 |     0.8122 | 14.7h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_32/imagenet2012/)          |
| imagenet21k | ViT-L_32     | imagenet2012 |     0.812  | 14.7h             | [tensorboard.dev](https://tensorboard.dev/experiment/vNVL9RFmTBKJ4uK81CbGMQ/#scalars&_smoothingWeight=0&regexInput=imagenet21k/ViT-L_32/imagenet2012/)          |

We also would like to emphasize that high-quality results can be achieved with
shorter training schedules and encourage users of our code to play with
hyper-parameters to trade-off accuracy and computational budget.
Some examples for CIFAR-10/100 datasets are presented in the table below.

| upstream    | model    | dataset      | total_steps / warmup_steps  | accuracy | wall-clock time |                                                                         link |
| ----------- | -------- | ------------ | --------------------------- | -------- | --------------- | ---------------------------------------------------------------------------- |
| imagenet21k | ViT-B_16 | cifar10      | 500 / 50                    |   0.9859 |             17m | [tensorboard.dev](https://tensorboard.dev/experiment/QgkpiW53RPmjkabe1ME31g/) |
| imagenet21k | ViT-B_16 | cifar10      | 1000 / 100                  |   0.9886 |             39m | [tensorboard.dev](https://tensorboard.dev/experiment/w8DQkDeJTOqJW5js80gOQg/) |
| imagenet21k | ViT-B_16 | cifar100     | 500 / 50                    |   0.8917 |             17m | [tensorboard.dev](https://tensorboard.dev/experiment/5hM4GrnAR0KEZg725Ewnqg/) |
| imagenet21k | ViT-B_16 | cifar100     | 1000 / 100                  |   0.9115 |             39m | [tensorboard.dev](https://tensorboard.dev/experiment/QLQTaaIoT9uEcAjtA0eRwg/) |

## Bibtex

```
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

**This is not an official Google product.**
