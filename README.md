# Neural Networks

A personal learning repository for understanding neural networks from the ground up.


## Resources

- [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Daniel Bourke - Learn Pytorch for deep learning](https://youtu.be/Z_ikDlimN6A?si=8mKcCGbLTdbcQUtg)

## Topics Covered

| # | Topic | Notebook | Description |
|---|-------|----------|-------------|
| 1 | **PyTorch Fundamentals** | `notebook/pytorch-fundamentals/fundamentals.ipynb` | Tensors, operations, and core PyTorch concepts |
| 2 | **PyTorch Workflow** | `notebook/pytorch-fundamentals/pytorch_workflow.ipynb` | End-to-end model building, training, and saving/loading |
| 3 | **Derivatives** | `notebook/backpropgation/derivatives.ipynb` | Understanding derivatives for backpropagation |
| 4 | **Manual Backpropagation** | `notebook/backpropgation/manual_backpropgation.ipynb` | Hand-computing gradients through a network |
| 5 | **Backprop with Neurons** | `notebook/backpropgation/manual_backprogation_neuron.ipynb` | Backpropagation applied to individual neurons |
| 6 | **Building a NN Library** | `notebook/backpropgation/Building_nn_lib.ipynb` | Implementing a neural network library from scratch |
| 7 | **Bigram Language Model** | `notebook/bigram-make-more/building_make_more.ipynb` | Character-level bigram model on a names dataset |
| 8 | **MLP Language Model** | `notebook/bigram-make-more-mlp/mlp.ipynb` | Multi-layer perceptron for character-level prediction |
| 9 | **Backprop Ninja** | `notebook/backpropninja/build_makemore_backprop ninja.ipynb` | Manual backprop through the MLP (exercises) |
| 10 | **WaveNet** | `notebook/wavenet/wavenet.ipynb` | WaveNet-style deep network with dilated causal structure |
| 11 | **nano-GPT (Bigram)** | `notebook/nano-gpt/biagram.py` | Bigram language model on Tiny Shakespeare with training loop & text generation |

## Key Concepts Explored

- **Autograd & Gradients** — building intuition by computing gradients manually before using PyTorch autograd
- **Weight Initialization** — understanding how initialization affects training (tanh saturation, etc.)
- **Batch Normalization** — what it does, why it helps, and a from-scratch implementation
- **Character-level Language Models** — bigram → MLP → WaveNet progression on a 32K names dataset
- **GPT-style Text Generation** — bigram language model trained on Tiny Shakespeare with autoregressive generation
- **Building from Scratch** — implementing `Value`, `Neuron`, `Layer`, and `MLP` classes by hand

## Repository Structure

```
neural_networks/
├── notebook/
│   ├── pytorch-fundamentals/          # PyTorch fundamentals & workflow
│   ├── backpropgation/             # Derivatives & backprop from scratch
│   ├── backpropninja/              # Manual backprop exercises
│   ├── bigram-make-more/           # Bigram character-level model
│   ├── bigram-make-more-mlp/       # MLP character-level model
│   ├── wavenet/                    # WaveNet architecture
│   └── nano-gpt/                   # GPT-style bigram language model
├── models/                         # Saved model weights
└── LICENSE
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/iharsh02/neural_networks.git
cd neural_networks

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch (pick the right command for your OS/CUDA version)
# https://pytorch.org/get-started/locally/

# Install other dependencies
pip install -r requirements.txt

# Launch the notebooks
jupyter notebook
```