# Configs

This directory contains `ml_collections.ConfigDict` configurations. It is
structured in a way that factors out common configuration parameters into
`common.py` and model configurations into `models.py`.

To select one of these configurations you can specify it on the command line:

```sh
python -m vit_jax.main --config=$(pwd)/vit_jax/configs/vit.py:b32,cifar10
```

The above example specifies the additional parameter `b32,cifar10` that is
parsed in the file `vit.py` and parametrizes the configuration.

Note that you can override any configuration parameters at the command line by
specifying additional parameters like `--config.accumulation_steps=1`.
