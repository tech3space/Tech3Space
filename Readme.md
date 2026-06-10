# 🧠 Tech3Space Research: Deep Dive into Transformer Architectures

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Transformers-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NLP-LLMs-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Tech3Space-Research-orange?style=for-the-badge"/>
</p>

## 🚀 Introduction

Transformers have revolutionized Artificial Intelligence by replacing recurrent architectures with **self-attention mechanisms**, enabling efficient parallel computation and superior long-range dependency modeling.

The original Transformer was introduced in the paper:

> **"Attention Is All You Need" (2017)**

Today, almost every modern Large Language Model (LLM)—including GPT, LLaMA, Mistral, Gemma, Qwen, Falcon, and Mixtral—is built upon Transformer principles with architectural innovations.

---

# 🏗️ Complete Transformer Architecture

```text
                Input Tokens
                      │
                      ▼
             Token Embedding Layer
                      │
                      ▼
        Positional Encoding / RoPE
                      │
                      ▼
        ┌───────────────────────────┐
        │      Transformer Block     │
        │                           │
        │   ┌───────────────────┐   │
        │   │   RMSNorm/LN      │   │
        │   └────────┬──────────┘   │
        │            ▼              │
        │   Multi-Head Attention    │
        │            │              │
        │      Residual Add         │
        │            ▼              │
        │     RMSNorm / LayerNorm   │
        │            ▼              │
        │ Feed Forward (MLP/GLU)    │
        │            │              │
        │      Residual Add         │
        └───────────────────────────┘
                      │
                Repeat N Layers
                      │
                      ▼
             Final Normalization
                      │
                      ▼
               Linear Projection
                      │
                      ▼
                 Softmax Output
```

---

# 📚 Mathematical Foundations

## 1. Token Embedding

Each token is mapped into a dense vector:

[
x_i = E(token_i)
]

where

* (E) = embedding matrix
* (x_i \in \mathbb{R}^{d_{model}})

---

## 2. Positional Encoding

Original Transformer:

[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
]

[
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
]

Modern LLMs instead prefer **Rotary Position Embeddings (RoPE)**.

---

## 3. Query, Key and Value

[
Q=XW_Q
]

[
K=XW_K
]

[
V=XW_V
]

where

* (W_Q,W_K,W_V) are learned matrices.

---

## 4. Scaled Dot-Product Attention

[
Attention(Q,K,V)
================

Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
V
]

This computes similarity between tokens and aggregates contextual information.

---

## 5. Multi-Head Attention

[
head_i
======

Attention(Q_i,K_i,V_i)
]

[
MHA
===

Concat(head_1,\dots,head_h)W_O
]

Multiple heads learn different relationships simultaneously.

---

## 6. Residual Connection

[
y=x+f(x)
]

Residual pathways improve optimization and stabilize deep networks.

---

## 7. Layer Normalization

For vector (x):

[
LN(x)
=====

\gamma
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
]

Used heavily in the original Transformer and GPT family.

---

## 8. RMSNorm

Modern LLMs often replace LayerNorm with RMSNorm:

[
RMS(x)
======

\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
x_i^2
}
]

[
RMSNorm(x)
==========

\frac{x}
{RMS(x)+\epsilon}
\cdot g
]

Advantages:

* Faster computation
* Lower memory overhead
* Similar or improved stability

---

## 9. Feed Forward Network (FFN)

Original Transformer:

[
FFN(x)
======

W_2
(
GELU(W_1x)
)
]

Usually expands dimensions by 4× before projection back.

---

# ⚡ Modern Activation Functions

## GELU

[
GELU(x)
=======

x
\Phi(x)
]

Smooth alternative to ReLU, widely used in GPT.

---

## SwiGLU

[
SwiGLU(x)
=========

Swish(xW_1)
\odot
(xW_2)
]

Provides stronger expressive power and is common in LLaMA and Mistral.

---

## GeGLU

[
GeGLU(x)
========

GELU(xW_1)
\odot
(xW_2)
]

Used in models such as Gemma.

---

# 🧩 Encoder vs Decoder

| Feature                   | Encoder       | Decoder         |
| ------------------------- | ------------- | --------------- |
| Bidirectional             | ✅             | ❌               |
| Masked Attention          | ❌             | ✅               |
| Reads future tokens       | ✅             | ❌               |
| Autoregressive generation | ❌             | ✅               |
| Typical use               | Understanding | Text generation |

Examples:

* **Encoder-only:** BERT
* **Decoder-only:** GPT, LLaMA, Mistral
* **Encoder-Decoder:** T5, BART

---

# 🔥 Transformer Block (Pre-Norm)

```text
Input
  │
  ▼
RMSNorm / LayerNorm
  │
  ▼
Self-Attention
  │
  ▼
Residual Add
  │
  ▼
RMSNorm
  │
  ▼
Feed Forward Network
  │
  ▼
Residual Add
  │
Output
```

---

# 📊 Comparison of Modern Transformer Models

| Model   | Main Innovation            | Normalization       | Activation    | Attention Type      | Positional Encoding | Other Key Features                 |
| ------- | -------------------------- | ------------------- | ------------- | ------------------- | ------------------- | ---------------------------------- |
| GPT     | Original Decoder-Only      | LayerNorm           | GELU          | Multi-Head          | Learned Absolute    | Basic Transformer                  |
| LLaMA   | RMSNorm + SwiGLU           | RMSNorm             | SwiGLU        | Multi-Head          | RoPE                | Pre-Norm, large FFN                |
| Falcon  | Multi-Query Attention      | LayerNorm           | GELU          | Multi-Query (MQA)   | RoPE                | Faster inference                   |
| Mistral | Sliding Window + GQA       | RMSNorm             | SwiGLU        | Grouped-Query (GQA) | RoPE                | Sliding Window Attention           |
| Mixtral | Mixture of Experts         | RMSNorm             | SwiGLU        | GQA                 | RoPE                | Sparse MoE (8 experts, 2 active)   |
| Qwen    | Long Context + Dynamic NTK | RMSNorm             | SwiGLU        | GQA                 | RoPE + Dynamic      | Excellent long-context scaling     |
| Phi     | Small High-Quality Models  | LayerNorm / RMSNorm | GELU / SwiGLU | Multi-Head          | RoPE                | Focus on curated training data     |
| Gemma   | Lightweight + Efficient    | RMSNorm             | GeGLU         | Multi-Head / GQA    | RoPE                | Optimized for efficient deployment |

---

# 🧠 Key Architectural Innovations

### GPT

* Decoder-only Transformer
* LayerNorm + GELU
* Learned positional embeddings
* Autoregressive next-token prediction

### LLaMA

* RMSNorm replaces LayerNorm
* SwiGLU feed-forward blocks
* RoPE positional encoding
* Pre-normalization improves training stability

### Falcon

* Introduces Multi-Query Attention (MQA)
* Shares Keys/Values across heads
* Reduces memory and improves inference speed

### Mistral

* Grouped-Query Attention (GQA)
* Sliding Window Attention for efficient long sequences
* RoPE positional encoding

### Mixtral

* Sparse Mixture of Experts (MoE)
* Activates only a subset of experts per token
* Increases capacity without proportional compute cost

### Qwen

* Dynamic NTK-aware RoPE scaling
* Strong long-context capabilities
* Uses GQA with SwiGLU and RMSNorm

### Phi

* Compact models trained on carefully curated data
* Demonstrates that data quality can compensate for model size

### Gemma

* Lightweight architecture optimized for efficient deployment
* Uses RMSNorm, GeGLU, and RoPE
* Designed for strong performance with practical resource usage

---

# 🎯 Core Takeaways

* **Self-Attention** is the foundation of Transformer models.
* **RMSNorm** has become the preferred normalization in many modern LLMs.
* **RoPE** largely replaces absolute positional embeddings for better extrapolation.
* **SwiGLU** and **GeGLU** improve feed-forward expressiveness over GELU alone.
* **GQA** and **MQA** reduce memory usage while maintaining strong performance.
* **Mixture of Experts (MoE)** scales model capacity efficiently by activating only selected experts.
* Architectural refinements across GPT, LLaMA, Falcon, Mistral, Mixtral, Qwen, Phi, and Gemma illustrate the rapid evolution of Transformer design.

---

<p align="center">
<b>⭐ Tech3Space Research Series</b><br/>
Exploring Deep Learning, Large Language Models, Transformers, and Modern AI Architectures.
</p>
