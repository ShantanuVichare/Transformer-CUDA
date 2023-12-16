# Enhancing Large Language Model Performance with C++/CUDA Extensions in PyTorch

## Abstract

PyTorch Machine Learning (ML) implementations often face performance challenges due to Python interpreter overhead, leading to fragmented kernel launches. To address this, leveraging C++ extensions in PyTorch can enhance execution times. The project focused on optimizing the Self Attention layer, critical in Large Language Models (LLMs), by implementing it in low-level C++/CUDA code.
CUDA extensions for PyTorch, demonstrated by benchmarks, showcased a ~30% improvement over PyTorch/Python implementations for a simple LSTM unit, indicating potential benefits for other constructs like a transformer layer.
For SelfAttention, we were able to get ~15% improvement on the lower range of sequence length and embedding sizes. We expect this to only improve with custom kernels and further optimizations.

## Contents

1. [Problem statement](#1-problem-statement)
2. [Solution description](#2-solution-description)
3. [Overview of results](#3-overview-of-results-demonstration-of-your-project)
4. [Install and Build steps](#4-install-and-build-steps)
5. [Conclusions and Future Work](#5-conclusions-and-future-work)
6. [References](#references)

## 1. Problem statement

PyTorch implementations of Machine Learning (ML) models often bear the overhead introduced by the context switching introduced by the Python interpreter. This leads to fragmented kernel launches and reduced performance. This gives room for improving performance via implementing high-level concepts in low-level programming constructs and thus mitigating the overhead.

PyTorch supports a way to integrate C++ extensions which can inherently implement customized behaviors and thus speeding up execution (training and inference) times. In this project, we targeted a Self Attention layer which forms the basic building block of all modern LLMs.

CUDA extensions for PyTorch have demonstrated potential in improving performance over its naïve PyTorch implementations. For reference, we have some benchmarks from the official [CUDA extensions resource](https://pytorch.org/tutorials/advanced/cpp_extension.html#cuda-extensions) :

| Implementation             | Forward pass execution time (μs) | Backward pass execution time (μs) |
| -------------------------- | -------------------------------- | --------------------------------- |
| PyTorch/Python on GPU      | 187.719                          | 410.815                           |
| CUDA extension for PyTorch | 129.431                          | 304.641                           |

Since these performance improvements of ~30% have been demonstrated for a simple LSTM unit, we can
expect these benefits to translate into implementations of other high-level constructs like a transformer layer in low-level C++/CUDA code.

## 2. Solution description

Coming to the solution involved several key steps before the implementation could even start.

1. Learning more about PyTorch’s internal APIs and its integration with the Torch C++ API was crucial for writing the C++ extension.
2. Understanding PyTorch’s implementation of Optimizer helped in figuring out the forward and backward pass integrations especially in the linker code.
3. Computational graph analysis of the SelfAttention layer was necessary to modularize the implementation and write its lower-level implementation in C++.
4. Mathematical derivation of gradients across the computational graph was necessary in order to implement the backward pass in C++. PyTorch’s execution engine typically abstracts away this by implementing all these gradients at the machine code layer itself.

Multi-head Self Attention was implemented as part of this project. Its PyTorch-based implementation can be found in `self_attention/python/SelfAttention.py` whereas the ATen framework-based implementation can be found in `self_attention/extended/self_attention.cpp`. This was compiled as a C++ binary which was linked to PyTorch using setuptools packaging and alternatively Just-In-Time (JIT) compilation.

The ATen framework functions as a wrapper over low-level architecture-specific primitives allowing us to write highly compatible code to work with the PyTorch framework. The testing machine has a Nvidia GPU (GTX 1050) Pascal architecture with 4GB of graphics memory. In order to give a perspective on GPU occupation, a test case of batch size=8, sequence length=10, attention heads=4, and embedding size=64, occupied about 35% of the available onboard memory.

Computational graph analysis was done to understand the flow of data and gradient calculations.

<img width="700" alt="self-attention-computation-graph" src="https://github.com/ShantanuVichare/Transformer-CUDA/assets/62808114/ee8494cc-0d76-4ce8-bfd7-289bdf4f8384">

Additionally, mathematical formulations for the derivatives were calculated for each variable node and the derivative of stable Softmax gradient was derived.

## 3. Overview of results

- The C++ implementation was able to match the PyTorch implementation’s accuracy exactly.
- On average, the C++ implementation outperforms the PyTorch implementation for embedding sizes upto 512 and sequence lengths upto 16.
- Scaling analysis reveals that PyTorch’s implementation is able to scale better for larger input sizes, possibly due to better hyper parameter control
- Backward pass performance of PyTorch is consistently superior as was expected. This is due to the fact that PyTorch’s automatic differentiation engine can automatically parallelize computation graphs, may use a more efficient flow of operations overall, and is also implemented in C++.

## 4. Install and Build steps:

- (Optional) Create Python environment using venv or conda.
- `cd` into `self_attention/` directory
- Setup required packages using `pip install -r requirements.txt`.
- Build extensions by going into the `extended/` folder and executing `python setup.py install --user`
- JIT-compile extensions by going into the `extended/` folder and calling `python jit.py`, which will JIT-compile the extension and load it
- Run tests and generate plots using `test.ipynb`

## 5. Conclusions and Future Work

While this project focuses on several PyTorch internals and its adaptor patterns to leverage high performance computing, there are several things that can be worked on in the future including:

- Writing a custom CUDA kernel to further reducing kernel launches by kernel fusing and improve memory access patterns.
- We can use CUDA streams to parallelize computation graphs in our custom kernel, effectively optimizing the backward pass.
- Scaling to Transformer node can be improved by identifying performance bottlenecks and defining additional hyperparameters for making use of additional flexibility offered by custom kernels like templatizing.

## References

1. [PyTorch docs: Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html#cuda-extensions)
2. [PyTorch FAQ](https://pytorch.org/docs/stable/notes/faq.html)
3. [PyTorch C++/CUDA Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
4. [Autograd fundamentals](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
5. Gradient derivatives
   - [Getting derivatives from implicit functions with autograd](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Getting%20derivatives%20from%20implicit%20functions%20with%20autograd.pdf)
   - [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
   - [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
