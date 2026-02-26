# Deep Learning Fundamentals

An introduction to deep learning for programmers, focused on practical implementation with PyTorch.

## Learning Path

### Main Track

Work through these notebooks in order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Introduction to Neural Networks](01_intro_neural_networks.ipynb) | History, perceptron, MLPs, activation functions, training concepts, architectures overview |
| 2 | [PyTorch Basics](02_pytorch_basics.ipynb) | Tensors, GPU usage, `nn.Module`, `nn.Sequential`, building models |
| 3 | [Your First Neural Network](03_first_neural_network.ipynb) | End-to-end FashionMNIST classification: data loading, model, training loop, evaluation |
| 4 | [Training Dynamics](04_training_dynamics.ipynb) | Loss functions, optimizers, regularization (L2, Dropout), learning rate scheduling |

### Complementary Material

Optional deep-dives referenced from the main track:

| Resource | Description |
|----------|-------------|
| [XOR & Linear Separability](complementary/xor_linearity.ipynb) | Why linear models fail on XOR, feature engineering vs. representation learning, universal approximation theorem |
| [Gradient Descent from Scratch](complementary/gradient_descent_from_scratch.ipynb) | ADALINE in NumPy, 3D cost landscape, comparison with PyTorch autograd |
| [TensorFlow Playground Guide](complementary/tf_playground_guide.ipynb) | Hands-on experiments with [TF Playground](https://playground.tensorflow.org/) + equivalent PyTorch code |

## Prerequisites

- Python programming experience
- Basic linear algebra (vectors, matrices)
- PyTorch installed (`pip install torch torchvision`)
