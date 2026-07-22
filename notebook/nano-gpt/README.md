# nano-GPT

A character-level GPT-style transformer language model trained on [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1MB of Shakespeare text). Built following [Andrej Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture.

## Architecture

```
Input Tokens
    │
    ▼
┌─────────────────────────┐
│  Token Embedding Table  │  (vocab_size × n_embed)
│  + Position Embedding   │  (block_size × n_embed)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Transformer Block    │ ×n_layer
│  ┌───────────────────┐  │
│  │   LayerNorm       │  │
│  │   Multi-Head      │  │
│  │   Self-Attention   │  │
│  │   + Residual      │  │
│  ├───────────────────┤  │
│  │   LayerNorm       │  │
│  │   FeedForward     │  │
│  │   (4× expansion)  │  │
│  │   + Residual      │  │
│  └───────────────────┘  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Final LayerNorm        │
│  Linear → vocab_size    │
└────────────┬────────────┘
             │
             ▼
        Next Token
```

## Concepts Learned

### Self-Attention
- **Scaled dot-product attention** with key (K), query (Q), and value (V) projections
- **Causal masking** using a lower-triangular matrix to prevent attending to future tokens
- Scaling by `head_size^(-0.5)` (i.e. `1/sqrt(d_k)`) so attention logits stay roughly unit-variance and softmax doesn't saturate / become too peaky

### Multi-Head Attention
- Running **multiple attention heads in parallel**, each with `head_size = n_embed // num_heads`
- Concatenating outputs along the **feature dimension** (`dim=-1`), not the time dimension
- **Projection layer** to map concatenated heads back into the residual stream

### Feedforward Network
- Position-wise FFN with **4× inner expansion**: `n_embed → 4*n_embed → n_embed`
- Gives the model capacity to "think" (compute) after communication (attention)
- Uses ReLU activation between the two linear layers

### Transformer Block
- **Communication** (self-attention) followed by **computation** (feedforward)
- **Pre-norm** architecture: LayerNorm is applied *before* attention and FFN, not after
- **Residual connections**: skip connections so gradients flow through deep networks

### Regularization
- **Dropout** applied to attention weights, projection layers, and FFN outputs
- Prevents overfitting, especially important when scaling up model size

### Positional Embeddings
- **Learned position embeddings** (not sinusoidal) added to token embeddings
- Allows the model to understand token order within the context window

### Text Generation
- **Autoregressive sampling**: predict next token, append to context, repeat
- Context is cropped to `block_size` to stay within the positional embedding range

## Model Progression

The model was built incrementally, with each step improving val loss:

| Stage | Description | Val Loss |
|-------|-------------|----------|
| Bigram baseline | Simple token embedding → logits | ~2.50 |
| + Self-Attention | Single head of self-attention | ~2.40 |
| + Multi-Head | 4 heads (head_size=8, n_embed=32) | ~2.27 |
| + FeedForward | FFN after attention | ~2.24 |
| + Blocks + LayerNorm | 4 stacked transformer blocks | ~2.20 |
| Scaled up | 6 layers, 384 embed, 6 heads, dropout | ~1.49 (needs re-measuring after the residual/scaling fixes — earlier number predates them) |

## Hyperparameters

```python
batch_size = 64       # parallel sequences per step
block_size = 256      # maximum context length
max_iters = 5000      # training iterations
eval_interval = 500   # evaluate every N steps
learning_rate = 3e-4  # AdamW learning rate
n_embed = 384         # embedding dimension
n_layer = 6           # number of transformer blocks
num_heads = 6         # attention heads per block
dropout = 0.2         # dropout rate
```

**Parameter count**: ~10.8M

## Usage

```bash
# Download the dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train and generate
python biagram.py
```

Training takes **~40-45 minutes on an RTX 3050** with the full config.

## Key Bugs Encountered & Fixed

1. **`__init__` not accepting `vocab_size`** — constructor used global `vocab_size` directly but was called with an argument
2. **Multi-head concat on wrong dimension** — `dim=1` (time) instead of `dim=-1` (features), causing shape mismatch with `lm_head`
3. **Missing hyperparameters** — `n_layer`, `num_heads`, and `dropout` needed to be defined in the config
4. **Keyword argument typo** — `num_head=` instead of `num_heads=` when instantiating `Block`
