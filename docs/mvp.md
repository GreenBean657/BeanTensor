# BeanTensor MVP scope decisions

This document records the scope boundaries for the BeanTensor minimum viable product. Later work—hyperparameters, accuracy targets, and kernel build choices—builds on these decisions.

- **Python bindings:** Out of the MVP. BeanTensor ships as a C++ library; Python bindings are a separate phase.
- **Activations:** ReLU only.
- **Weight initialization:** He (Kaiming) uniform, fan-in.
- **Accuracy target:** The PyTorch comparison harness result on the same architecture, minus a one-percentage-point margin.
- **Model topology:** A single MLP; convolutions are not in the MVP.

These decisions live in `docs/` and never in `README.md`; `README.md` does not duplicate project decisions.
