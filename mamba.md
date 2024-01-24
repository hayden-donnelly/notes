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
h^\prime &= Ah(t) + Bx(t) \tag{1a}\\ 
h_t &= \bar{A}h_{t-1} + \bar{B}x_t \tag{2a}\\ 
\bar{K} &= (C\bar{B}, C\bar{AB}, ... , C\bar{A}^k\bar{B}, ...) \tag{3a}\\
y(t) &= Ch(t) \tag{1b}\\ 
y_t &= Ch_t \tag{2b}\\
y &= x * \bar{K} \tag{3b}\\
\end{align}
$$
```

### Discretization
- The first stage transforms the "continuous parameters" $(\Delta, A, B)$ to "discrete parameters" 
$(\bar{A}, \bar{B})$ through fixed formulas.
- These are $\bar{A} = f_{A}(\Delta, A)$ and $\bar{A} = f_{B}(\exp(\Delta A) - I) \cdot \Delta B$.
- The pair $(f_{A}, f_{B})$ is called a discretization rule.
- Various rules can be used such as the zero-order hold (ZOH) defined in equation 4.

```math
$$
\begin{align}
\bar{A} &= \exp(\Delta A) \tag{4a}\\
\bar{B} &= (\Delta A)^{-1}\exp(\Delta A) - I) \cdot \Delta B \tag{4b}\\
\end{align}
$$
```

- Discretization has deep connections to continuoous-time systems which can give them additional properties
such as resolution invariance and automatic normalization.
- It also has connections to the gating mechanisms of RNNs.

### Computation
- After discretization the model can be computed either as a linear recurrence or a global convolution.
- Usually the model uses convolutional mode for efficient parallelizable training and is switched
into recurrent mode for efficient autoregressive inference.

### Linear Time Invariance (LTI)
- An important property of equations 1-3 is that the model's dynamics are constant through time.
- This means that $(\Delta, A, B, C)$ and consequently $(\bar{A}, \bar{B})$ are fixed for all time-steps.
- This is called linear time invariance (LTI).
- Informally LTI SSMs can be thought of as equivalent to any linear recurrence (2a) or convolution (3b).
- LTI is used as an umbrella term for these classes of models.
- So far all structured SSMs have been LTI because of efficiency constraints.
- LTI models have fundamental limitations in modeling certain types of data.
- The authors remove the LTI constraint and overcome the efficiency bottlenecks.

### Structure and Dimensions
- Structured SSMs get their name from the fact that computing them efficiently requires imposing structure
on the $A$ matrix.
- The most popular form of structure is diagonal, which the authors also use.
- $B$ and $C$ are both essentially vectors (one of their dimensions is 1) so it's easy to see how 
they can be represented by $N$ numbers (they are $N$ dimensional vectors). Since $A$ is diagonal,
and there are $N$ entries on the diagonal, it can also be represented by $N$ numbers.
- In the case of diagonal structure, the $A\in\mathbb{R}^{N\times N}$, $B\in\mathbb{R}^{N\times 1}$, 
$C\in\mathbb{R}^{1\times N}$, matrices can all be represented by $N$ numbers.
- To operate over an input sequence $x$ of batch size $B$ and length $L$ with $D$ channels, the SSM is
applied independently to each channel.
- In this case, the total hidden state has dimension $DN$ per input, and computing over the sequence length
requires $O(BLDN)$ time and memory; this is the root of the efficiency bottleneck.

### General State Space Models
- The term "state space model" has a very broad meaning, it simply represents the notion of any recurrent
process with a latent state.
- It has been used to refer to Markov decision processes (MDP), dynamic causal modelling (DCM) 
hidden Markov models (HMM) and linear dynamical systems (LDS), and recurrent models at large.
- The authors use the term SSM to refer exclusively to the class of structured SSMs or S4 models.

### SSM Architectures
- Linear attention is an approximation of self-attention involving a recurrence which can be viewed as a
degenerate linear SSM.
- H3 generalized this recurrence to use S4; it can be viewed as an architecture with an SSM sandwhiched 
by two gated connections. H3 also inserts a standard local convolution, which they frame as a shift-SSM,
before the main SSM layer.
- Hyena uses the same architecure as H3 but replaces the S4 layer with an MLP-parameterized global
convolution.
- RetNet adds an additional gate to the architecture and uses a simpler SSM, allowing an alternative
parallelizable computation path, using a variant of multi-head attention (MHA) instead of convolutions.
- RWKV is an RNN designed for language modelling based on another linear attention approximation.
Its main "WKV" mechanism involves LTI recurrences and can be viewed as the ratio of two SSMs.

## Selective State Space Models
- Time-varying SSMs cannot use convolutions, this presents a technicall challenge of how to compute them
efficiently.
- The authors overcome this with a hardware-aware algorithm that exploits the memory hierarchy on modern
hardware.

### Motivation: Selection as a Means of Compression
- The authors argue that a fundamental problem of sequence modeling is compressing context into a smaller 
state.
- The tradeoffs of popular sequence models can be viewed from this point of view.
- Attention is both effective and inefficient because it explicitly does not compress context at all.
- Storing the entire context (i.e. the KV cache) directly causees the slow linear-time inference
and quadratic-tim training of transforms.
- On the other hand, recurrent models are efficient because they have a finite state, but their
effectiveness is limited by how well this state had compressed the context.
- To understand this principle, the authors focus on two synthetic tasks:
    - The Selective Copying task which requires content-aware reasoning to be able to memorize the relevant
    tokens and filter out the irrelevant ones.
    - The Induction Heads task which requires context-aware reasoning to know when to produce the correct
    output in the appropriate context.
- These tasks reveal the failure mode of LTI models.
- From the recurrent view, their constant dynamics cannot let them select the correct information from their
context, or affect the hidden state passed along the sequence in an input-dependent way.
- From the convolutional view, convolutions have difficulty with the Selective Copying task because the
spacing between inputs-to-otuputs is varying and cannot be modeled by static convolution kernels.
- The efficiency vs effectiveness tradeoff: efficient models must have a small state, while effective models
must have a state that contains all the necessary information from the context.

## Improving SSMs with Selection
- One method of incorporating a selectino mechanism into models is by letting their parameters that affect
interactions along the sequence (e.g. the recurrent dunamics of an RNN or the convolution kernel of a CNN)
be input-dependent.
- Algorithm 1, SSM (S4):
```python
x = input(shape=(B, L, D))
A = param(shape=(D, N)) # Structured N x N matrix.
B = param(shape=(D, N))
C = param(shape=(D, N))
delta = tau_delta(param(shape=(D)))
A_bar, B_bar = discretize(delta, A, B) # shape=(D, N)
y = SSM(A_bar, B_bar, C)(x)
return y
```
- Algorithm 2, SSM + Selection (S6):
```python
x = input(shape=(B, L, D))
A = param(shape=(D, N))
B = s_B(shape=(B, L, N))(x)
C = s_C(shape=(B, L, N))(x)
delta = tau_delta(param(shape=(B, L, D)) + s_delta(shape=(B, L, D))(x))
A_bar, B_bar = discretize(delta, A, B) # shape=(B, L, D, N)
y = SSM(A_bar, B_bar, C)(x)
return y
```
- Algorithms 1 and 2 illustrate the main selection mechanism for Mamba, the main difference is making 
parameters $\Delta$, $B$, and $C$ functions of the input, along with the associated changes to tenor shapes.
- Importantly these parameters now have a length dimension $L$ which means that the model has changed from
time-invariant to time-varying.
- The authors specifically choose $s_{B}(x) = \text{Linear}_{N}(x)$, $s_{C} = \text{Linear}_{N}$, 
s_{\Delta}(x) = \text{Broadcast}_{D}(\text{Linear}_{1}(x))$, and $\tau_{\Delta} = \text{softplus}$, 
where \text{Linear}_{d} is a parameterized projection to dimension $d$.
- The choice of $s_{\Delta}$ and $\tau_{\Delta}$ is due to a connection to RNN gating mechanisms.
