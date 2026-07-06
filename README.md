# Neural Networks

A personal learning repository for understanding neural networks from the ground up

## Topics Covered

| # | Topic | Notebook | Description |
|---|-------|----------|-------------|
| 1 | **PyTorch Fundamentals** | `notebook/fundamentals.ipynb` | Tensors, operations, and core PyTorch concepts |
| 2 | **PyTorch Workflow** | `notebook/pytorch_workflow.ipynb` | End-to-end model building, training, and saving/loading |
| 3 | **Derivatives** | `notebook/backpropgation/derivatives.ipynb` | Understanding derivatives for backpropagation |
| 4 | **Manual Backpropagation** | `notebook/backpropgation/manual_backpropgation.ipynb` | Hand-computing gradients through a network |
| 5 | **Backprop with Neurons** | `notebook/backpropgation/manual_backprogation_neuron.ipynb` | Backpropagation applied to individual neurons |
| 6 | **Building a NN Library** | `notebook/backpropgation/Building_nn_lib.ipynb` | Implementing a neural network library from scratch |
| 7 | **Bigram Language Model** | `notebook/bigram-make-more/building_make_more.ipynb` | Character-level bigram model on a names dataset |
| 8 | **MLP Language Model** | `notebook/bigram-make-more-mlp/mlp.ipynb` | Multi-layer perceptron for character-level prediction |
| 9 | **Backprop Ninja** | `notebook/backpropninja/build_makemore_backprop ninja.ipynb` | Manual backprop through the MLP (exercises) |
| 10 | **WaveNet** | `notebook/wavenet/wavenet.ipynb` | WaveNet-style deep network with dilated causal structure |

## Key Concepts Explored

- **Autograd & Gradients** — building intuition by computing gradients manually before using PyTorch autograd
- **Weight Initialization** — understanding how initialization affects training (tanh saturation, etc.)
- **Batch Normalization** — what it does, why it helps, and a from-scratch implementation
- **Character-level Language Models** — bigram → MLP → WaveNet progression on a 32K names dataset
- **Building from Scratch** — implementing `Value`, `Neuron`, `Layer`, and `MLP` classes by hand

## Repository Structure

```
neural_networks/
├── notebook/
│   ├── fundamentals.ipynb          # PyTorch fundamentals
│   ├── pytorch_workflow.ipynb      # PyTorch model workflow
│   ├── backpropgation/             # Derivatives & backprop from scratch
│   ├── backpropninja/              # Manual backprop exercises
│   ├── bigram-make-more/           # Bigram character-level model
│   ├── bigram-make-more-mlp/       # MLP character-level model
│   └── wavenet/                    # WaveNet architecture
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

# Install dependencies
pip install torch numpy matplotlib jupyter

# Launch the notebooks
jupyter notebook
```