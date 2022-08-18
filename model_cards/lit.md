# Model Card: LiT (Locked image Tuning)

Last updated: 2022-06-19

Version: 1.0

- This doc: https://github.com/google-research/vision_transformer/blob/main/model_cards/lit.md
- Model Page: https://github.com/google-research/vision_transformer#lit-models
- Other Links:
  [LiT Blogpost](https://ai.googleblog.com/2022/04/locked-image-tuning-adding-language.html),
  [LiT Paper],
  [LiT Demo](https://google-research.github.io/vision_transformer/lit/)

A text/image input model that can be used to embed text/image individually,
and compute similarities between embeddings of text/image pairs. This enables
use cases like zero shot classification, or image/text retrieval.

Note that this model card refers to the models that have been released on
Github specifically (B16B_2, L16L). The [LiT Paper] also evaluates models that
have not been released and use different datasets for training. The Colab
[`lit.ipynb`] lists some more models (L16S, L16Ti) which are similar to L16L,
but with a smaller text tower.

[LiT Paper]: https://arxiv.org/abs/2111.07991
[`lit.ipynb`]: https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb

## Model Summary

- Architecture: Multimodal model with transformer text encoder and transformer
  image encoder.
- Inputs: Images presented in 224x224x3 input, text inputs are tokenized and
  cropped to the first 16 tokens.
- Outputs: Image and text embeddings (of size 768 or 1024).
- Person of contact: Andreas Steiner (Google Brain)
- Model authors: Xioahua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner,
  Daniel Keysers, Alexander Kolesnikov, Lucas Beyer (Google Brain)

Citation:

```bibtex
@article{zhai2022lit,
  title={LiT: Zero-Shot Transfer with Locked-image Text Tuning},
  author={Zhai, Xiaohua and Wang, Xiao and Mustafa, Basil and Steiner, Andreas and Keysers, Daniel and Kolesnikov, Alexander and Beyer, Lucas},
  journal={CVPR},
  year={2022}
}
```

## Model Data

Training data:

- [Pre-trained image-tower](http://arxiv.org/abs/2106.10270) (using the 
  recommended checkpoints from the paper, Section 4.2)
  - [ImageNet-21k](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)
- [BERT](http://arxiv.org/abs/1810.04805) pre-trained text tower
  - [BookCorpus](https://github.com/jackbandy/bookcorpus-datasheet)
  - English wikipedia
- Multi-modal datasets
  - [CC12M](https://arxiv.org/abs/2102.08981)
  - [YFCC100M](https://arxiv.org/abs/1503.01817)

Evaluation data (see also section [Evaluation Results](#evaluation-results)
below):

- Zero-shot classification
  - [ImageNet](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)
  - [ImageNet v2](http://arxiv.org/abs/1902.10811)
  - [CIFAR100](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
  - [Pets37](https://ieeexplore.ieee.org/abstract/document/6248092)
  - [Resisc45](http://arxiv.org/abs/1703.00121)
- Image-text retrieval
  - [MS-COCO Captions](https://arxiv.org/abs/1504.00325)

## Model Creation & Maintenance

The model has been initialized from BERT & ViT checkpoints (see details above
"training dataset"), and then contrastively tuned on CC12M and YFCC100M.

All datasets have been released in previous publications independent from this
model. The datasets and model are not regularly updated.

The published B16B_2 and L16L models are medium sized and can be used on a normal
computer, or on a single GPU/TPU.

| Model | B16B_2 | L16L |
| :--- | ---: | ---: |
| Size | 474 MB | 2.4 GB |
| Weights | 196M | 638M |
| Layers | 2x12 | 2x24 |
| Latency (single TPU core) | 1200/sec | 400/sec |

Software/hardware used for training:

- JAX 0.3.13, Flax 0.5.0
- 128 TPUv4 cores

Software/hardware used for deployment:

- JAX 0.3.13, Flax 0.5.0
- CPU/GPU/TPU

Compute requirements for training:

| Model | B16B_2 | L16L |
| :--- | ---: | ---: |
| Number of Chips	| 64 | 64 |
| Training Time (days) | 0.3 | 1 |
| Total Computation (FLOPS) | 2.7E+19 | 9E+19 |
| Measured Performance (TFLOPS/s) | 1153 | 1614 |
| Energy Consumption (MWh) | 0.14 | 0.16 |

Compute requirements for inference:

| Model | B16B_2 | L16L |
| :--- | ---: | ---: |
| FLOPS/example | approx. 10 | approx. 30 |

## Evaluation Results

Benchmark information:

- Zero-shot classification (as explained in [CLIP Paper])
  - We chose to evaluate a set of datasets that are commonly used, and provide
    insights where the model works very well (such as ImageNet v2 or CIFAR100),
    as well as where it is much more limited (such as Resisc45).
- Image-text retrieval (Appendix section I.3 in [LiT Paper])

[CLIP Paper]: https://arxiv.org/abs/2103.00020

Evaluation results:

| Model | B16B_2 | L16L |
| :--- | ---: | ---: |
| ImageNet zero-shot | 73.9% | 75.7% |
| ImageNet v2 zero-shot | 65.1% | 66.6% |
| CIFAR100 zero-shot | 79.0% | 80.5% |
| Pets37 zero-shot | 83.3% | 83.3% |
| Resisc45 zero-shot | 25.3% | 25.6% |
| MS-COCO Captions image-to-text retrieval | 51.6% | 48.5% |
| MS-COCO Captions text-to-image retrieval | 31.8% | 31.1% |

## Limitations

Known limitations:

- Any deployment of this model, both for commercial applications and
  non-commercial applications, is currently out of scope.
- Before using the model in a constrained (i.e. not deployed) environment, users
  should do in-depth testing for their specific use case (e.g. on a constrained
  set of class labels of interest).
- These models have only been trained on English text and will fail for most
  non-English inputs.
- These models have not been evaluated with respect to their biases and fairness
  aspects. We suspect that biases found in the datasets used for training will
  be replicated by model representations, and model predictions should a priori
  be considered to replicate these biases, with consequences to various fairness
  metrics.

Ethical considerations & risks:

- The publication is based on previous work ([CLIP Paper]) that has been shown
  (Section 7) to replicate gender biases, perform variably for different groups
  of people (by gender, skin color), and cause representational harm in varying
  degree for different groups of people (by age, skin color). In the same
  section, previous authors have shown that a discriminative image/text model
  has the potential to be used in a surveillance context for coarse
  classification (although not for fine-grained classification), potentially
  lowering the barrier for such problematic use cases.
- These models have not been evaluated for the problems mentioned in previous
  work, but until such an evaluation is performed, we expect similar risks.

## Model Usage

Sensitive use: The model has been trained on image datasets containing
pictures of people, both for the pre-training of the image encoder
(ImageNet-21k), and for the contrastive tuning (CC12M and YFCC100M).

The model is used exclusively in research for now:

- [Zero-Shot Text-Guided Object Generation with Dream Fields](https://arxiv.org/abs/2112.01455)
- [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

## Model Comparison

In comparison with "private data" model from [CLIP Paper]:

- As of 6/10/22, the best published CLIP model is the L/14-336px variant.
- Similar performance (e.g. ImageNet zero-shot classification accuracy:
  76.2% CLIP vs. LiT L16L 75.7%)
- LiT is trained solely on publicly available datasets, while CLIP is trained on
  a private undisclosed dataset.
- The LiT L16L model is considerably smaller: CLIP uses 576 tokens vs. LiT L16L
  uses 196 tokens – since the runtime/memory complexity of attention scales with
  the square of the number of tokens, this corresponds to a factor of 8.63x.

In comparison with "public data" model from [CLIP Paper]:

- The only model trained without the private data mentioned in the CLIP paper
  (Section D), namely on YFCC100M.
- LiT has much better performance (e.g. ImageNet zero-shot classification
  accuracy: 31.3% CLIP vs. LiT L16L 75.7%)

## System Dependencies

Can be used as a stand-alone model (e.g. for zero-shot classification or
retrieval), or as part of a more complex system (basically any system that uses
CLIP as a component can instead use a LiT model).

Pre-processing instructions can be found on Github:
[vit_jax/preprocess.py](https://github.com/google-research/vision_transformer/blob/main/vit_jax/preprocess.py).
The published models include a pre-processing configuration (specifying
tokenizer vocabulary and image pre-processing).

The model outputs image and text embeddings and a temperature. If similarities
are to be computed between image and text embeddings (e.g. for computing output
distributions), then the similarities between the embeddings should be computed
with the dot product, and these should then be multiplied by the temperature
before a softmax is applied.

## Changelog

- 2022-08-16: Replaced model B16B with an updated version B16B_2 that was
  trained for 60k steps (before: 30k) without linear head on the image side
  (before: 768) and has better performance.
