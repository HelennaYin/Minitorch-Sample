# MiniTorch (course project for Machine Learning Engineering)

This is a from-scratch autodiff and tensor library in Python inspired by PyTorch. It implements reverse-mode automatic differentiation, a Tensor type with core operations (including reductions and broadcasting), simple neural-network utilities (`Module`, `Parameter`), and basic optimizers. The codebase is structured for readability and tested with unit/property tests covering tensors, convolution, and NN behavior. 

## Features

- Reverse-mode **autodiff** with reverse topological backprop
- **Tensor** type with core ops (elementwise, reductions, broadcasting as implemented)
- **Modules & Parameters** for building small models
- Simple **optimizers** (e.g., SGD)
- **Tests** for tensors, convolution, and simple NN behavior

---

## Installation
pip install -U pip
pip install -r requirements.txt
pip install -Ue .


## Run Tests
pytest -q tests/test_tensor_general.py
pytest -q tests/test_conv.py
pytest -q tests/test_nn.py
