# Mamba: Linear Time Sequence Modeling with Selective State Spaces
[Paper](https://arxiv.org/abs/2312.00752)

## Abstract
- Subquadratic-time architecutes such as linear attention, gated convolution and recurrent models, 
and structured state space models (SSMs) have been developed to address the computational 
inefficiency of transformers but they have not performed as well on modalities such as language.
- Authors identify the inability to perform content based reasoning as a key weakness of these models.
- Letting SSM parameters be functions of the input addresses their weakness with discrete modalities.
- This allows the model to selectively propagate or forget information along the sequence length dimension.
- This change prevents the use of efficient convolutions.
- Authors design a hardware-aware parallel algorithm in recurrent mode to address this issue.
- Authors integreate selective SSMs into a model called Mamba.
- Mamba has no attention or MLP blocks.
- Mamba has 5x higher throughput than transformers and its scales linearly with sequence length.
- Performance improves on real data up to million-length sequences.
- Mamba achieves state-of-the-art performance across several modalities such as language, audio,
and genomics.
- On language modelling, Mamna-3B outperforms transformers of the same size and matches transformers
twice its size, both in pretraining and downstream evaluation.

## Introduction
- Structured state space sequence models (SSMs) have emerged as a promising class of architectures for
sequence modeling.
- These models can be interpreted as a combination of recurrent neural networks (RNNs) and convolutional
neural networks (CNNs) with inspiration from classical state space models.
- These mdoels can be computed very efficiently as either a recurrence or convolution, with linear or
near-linear scaling with respect to sequence length.
- They also have principled mechanisms for modeling long range dependencies in certain data modalities.
- SSMs have been successful in domains involving continuous signal data such as audio and vision.
- They tend to be less effective at modeling discrete and information-dense data like text.

## Selection Mechanism
- A key limitation of prior SSMs is the inability to efficiently select data in an input-dependent manner.
- Authors build on the intuition of synthetic tasks such as selective copy and induction heads to design
a simple selection mechanism.
- The selection mechanism parameterizes the SSM parameters based on the input.
- This allows the model to filter out irrelevant information and remember relevant information indefinitely.

## Hardware-aware Algorithm
- The new selection mechanism posese a technical challenge for the computation of the model.
- All prior SSMs must be time and input invariant in order to be computationally efficient.
- The authors overcome this with a hardware-aware algorithm that computes the model recurrently with a scan
instead of a convolution, but does not materialize the expanded state in order to avoid IO access between
different levels of the GPU memory hierarchy.
- The resulting implementation is faster than previous methods both in theory (scales linearly in sequence
length versus the pseudo-linearity of convolution-based SSMs) and on modern hardware (up to 3x faster on
A100 GPUs).

## Architecture
- Combination of SSM architectures and MLP block of transformers into a single block.
- Simple and homogenous architecture design incorporating selective state spaces.
- The Mamba architecture is fully recurrent with key properties that makes it suitable as a backbone for
foundational sequence models.
    1. Selectivity brings strong performance on dense modalities like language and genomics.
    2. Computation and memory scales linearly in sequence length during training, and unrolling the model
    during inference requires only constant time per step.
    3. The quality and efficiency yield performance improvements on real data up to a sequence length of 1M.
- Synthetic tasks such asa copying and induction heads have been propsed as being key to LLMs, Mamba
solves these easily and can extraploate solutions >1M tokens.
- Mamba out-performs prior SOTA models such as SaShiMi, Hyena and transformers on modeling audio waveforms
and DNA sequences, both in pretraining and downstream metrics. Its performance improves with longer context
in both settings.
- Mamba is the first linear-time sequence model to achieve transformer quality performance in language
modelling.
- Scaling laws up to 1B parameters show that Mamba exceeds the performance of a large range of baselines.

## State Space Models
- Structured state space sequence models (S4) are a recent class of sequence models that are broadly related
to RNNs and CNNs, and classical state space models.
- They are inspired by a particular continuous system that maps a 1-dimensional function or sequence 
$x(t)\in\mathbb{R}\rightarrow(t)\in\mathbb{R}$ through an implict latent state $h(t)\in\mathbb{R}^N$.
- Concretely, S4 models are defined with four parameters $(\Delta, A, B, C)$, which define a 
sequence-to-sequence transformation in two stages.

```math
$$
\begin{align}
h^\prime = Ah(t) + Bx(t) \text{1a} 
h_t = \bar{A}h_{t-1} + \bar{B}x_t \text{2a} 
\bar{K} = (C\bar{B}, C\bar{AB}, ... , C\bar{A}^k\bar{B}, ...) \text{3a} \notag\\
y(t) = Ch(t) \text{1b} 
y_t = Ch_t \text{2b}
y = x * \bar{K} \text{3b}
\end{align}
$$
```
