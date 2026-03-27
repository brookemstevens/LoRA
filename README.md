# LoRA: Low-Rank Adaptation of Large Language Models

## Overview

**LoRA** is a parameter-efficient fine-tuning technique for large pre-trained models. Rather than updating all weights during fine-tuning, LoRA injects small trainable matrices alongside frozen weights, reducing trainable parameters by orders of magnitude with no loss in performance.

---

## Why Fine-Tuning is Hard

Traditional fine-tuning of a model like GPT-3 (175B parameters) requires updating every weight in every matrix. This is a computationally enormous task. The learnable weight matrices in a Transformer include:

> **W_e, W_p, W_q, W_k, W_v, W_o, W_1, W_2, W_u, ...**

Each of these is a large matrix. In GPT-3, the attention matrices **W_q, W_k, W_v, W_o** are each of shape **(12,288 × 12,288)**. That's ~150 million parameters *each*.

Two classical approaches existed:
- **Full fine-tuning**: update all 175B parameters. Expensive.
- **Partial fine-tuning**: freeze most layers, update a subset. Less expressive.

LoRA introduces a third, far more efficient approach.

---

## The Key Insight: Low-Rank Decomposition

### Matrix Multiplication Shape Rule

When multiplying two matrices, the output takes the **outer dimensions**:

```
(m × r) × (r × n)  =  (m × n)
```

**Example:**

```
A            B              Output
[1000 × 4]  ×  [4 × 1000]  =  [1000 × 1000]
```

| Matrix         | Parameters       |
|----------------|------------------|
| (1000 × 1000)  | 1,000,000        |
| (1000 × 4) + (4 × 1000) | **8,000** |

That's a **125× reduction** in parameters. Shrink the rank to `r=2` and you get a **250× reduction**.

---


## GPT-3 Worked Example

For GPT-3's attention matrices (shape `12,288 × 12,288`) with rank `r = 4`:

| Method         | Parameters per matrix |
|----------------|-----------------------|
| Full fine-tuning | ~150,900,000        |
| LoRA (`r=4`)   | `2 × (12,288 × 4)` = **98,304** |

**Optimization improvement: ~1,535×** per attention matrix.

---

## The LoRA Method

Instead of directly modifying a weight matrix **W**, LoRA learns a low-rank *change* to it:

$$\Delta W = B \cdot A$$

where:
- **A** has shape `(r × k)`: projects *down* from input dimension `k` to rank `r`
- **B** has shape `(d × r)`: projects *back up* to output dimension `d`
- **r ≪ min(d, k)** (common choices: `r = 4` or `r = 8`)

For the square attention matrices in GPT-3, `d = k = d_model`. The full adapted weight used at inference is:

$$W' = W_0 + \Delta W = W_0 + B \cdot A$$

The output is additionally scaled by α/r, where α is a constant (typically set equal to the first `r` tried and left untuned). The modified forward pass is:

$$h = W_0 x + \frac{\alpha}{r} \cdot BAx$$

This scaling reduces the need to retune hyperparameters when varying `r`.

### Visual Diagram

```
Original weight matrix W₀  (d × k):
┌─────────────────────┐
│                     │   (frozen — never updated)
│        W₀           │
│                     │
└─────────────────────┘

LoRA decomposition  ΔW = B × A:

  A: (r × k)                    B: (d × r)
┌─────────────────────┐        ┌──┐
│                     │        │  │
│         A           │   ×    │  │
│                     │        │ B│
└─────────────────────┘        │  │
                               │  │
                               └──┘
  = ΔW: (d × k)  ← same shape as W₀

Combined:  W' = W₀ + BA
```

Because **W₀ + BA** is the same shape as **W₀**, there is **no additional inference cost** after training.

---

## LoRA Training Steps

1. **Freeze** all pretrained weights. Set W₀ and lock it.
2. **Inject** two small trainable matrices **A** (shape `r × k`) and **B** (shape `d × r`) alongside W₀.
3. **Initialize**: A is drawn from a Gaussian distribution, and B is initialized to **zero**. This means at the start of training:
   $$\Delta W = B \cdot A = \mathbf{0} \cdot A = \mathbf{0}$$
   The model begins training behaving exactly like the pretrained model.
4. **Train**: only A and B receive gradient updates. W₀ is untouched.
5. **Merge**: after training, compute the final weight:
   $$W' = W_0 + B \cdot A$$

---

## Why Does This Work?

The authors argue that when fine-tuning a pretrained model on a new task, the *necessary changes* to the weights live in a **low-dimensional subspace**. GPT-3 already encodes English grammar, world knowledge, and reasoning. Adapting it to write legal summaries only requires nudging a small slice of its behavior.

This is the same intuition behind **Principal Component Analysis (PCA)**: most of the useful variance in high-dimensional data can be captured in a surprisingly small number of dimensions. LoRA applies this idea directly to weight updates.

### Which Weights to Apply LoRA To?

The attention module contains four projection matrices: **W_q, W_k, W_v, and W_o**. The authors experimentally investigated which combinations to adapt under a fixed parameter budget (Table 5 of the paper). The key finding was that **adapting W_q and W_v together** yields the best overall performance (not all four matrices). Spreading the parameter budget across more matrix types (at a lower rank each) outperformed concentrating all parameters into a single matrix type. In the paper's experiments, LoRA is applied primarily to **W_q and W_v** for this reason.

---

## Performance

The authors benchmarked LoRA against full fine-tuning and other parameter-efficient methods. LoRA **matches or beats** full fine-tuning on downstream tasks with hundreds to thousands of times fewer trainable parameters.

**Limitations:**
- More complex domain shifts may require higher rank `r` or adapting more layers.
- The optimal choice of `r` and which layers to adapt is partly empirical; `r = 4` and `r = 8` are commonly used in industry.

---

## Appendix: Gradient Mathematics of LoRA

During training, standard backpropagation computes a loss **L** and propagates gradients backward through the network. In full fine-tuning, the gradient with respect to **W** is:

$$\frac{\partial \mathcal{L}}{\partial W}$$

This is a `(d × d)` matrix — expensive to store and update.

In LoRA, since W₀ is frozen, gradients only flow through **A** and **B**. Let the forward pass output be:

$$h = W_0 x + BAx$$

where A is `(r × k)` and B is `(d × r)`. Let `u = Ax` (shape `r × 1`), so `h_lora = Bu`. The gradient with respect to the output is **∂L/∂h**. By the chain rule:

### Gradient with respect to A:

$$\frac{\partial \mathcal{L}}{\partial A} = B^\top \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot x^\top$$

- **B^⊤** has shape `(r × d)`, collapsing the `(d×1)` upstream gradient into a `(r×1)` vector in the low-rank space.
- The outer product with **x^⊤** `(1 × k)` gives the result shape `(r × k)` — matching A.

### Gradient with respect to B:

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial h} \cdot (Ax)^\top = \frac{\partial \mathcal{L}}{\partial h} \cdot x^\top \cdot A^\top$$

- **∂L/∂h** has shape `(d × 1)` and **u^⊤ = (Ax)^⊤** has shape `(1 × r)`.
- The outer product gives result shape `(d × r)` — matching B.

### Why Initializing B = 0 Matters

At step 0, since B = 0:

$$\frac{\partial \mathcal{L}}{\partial A}\bigg|_{t=0} = \underbrace{B^\top}_{= \mathbf{0}} \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot x^\top = \mathbf{0}$$

A receives no gradient update at the very first step. Meanwhile, B's gradient `∂L/∂B = ∂L/∂h · (Ax)^T` is nonzero since A is initialized from a Gaussian. This means B begins accumulating updates first, after which A's gradients become nonzero and both matrices train together. Crucially, this initialization ensures `ΔW = BA = 0` at the start, so the model begins training behaving exactly like the pretrained model — a stable, well-defined starting point.

### Parameter Update Rule

Using standard gradient descent with learning rate η:

$$A \leftarrow A - \eta \cdot \frac{\partial \mathcal{L}}{\partial A}$$

$$B \leftarrow B - \eta \cdot \frac{\partial \mathcal{L}}{\partial B}$$

In practice, LoRA is compatible with any optimizer (Adam, AdamW, etc.), and the low dimensionality of A and B means the optimizer state itself is also dramatically smaller — a secondary but meaningful efficiency gain.

---

*Based on Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).*
