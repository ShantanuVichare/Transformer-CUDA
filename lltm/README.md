# C++/CUDA Extensions in PyTorch

Based on the example of writing a C++ extension for PyTorch [here](http://pytorch.org/tutorials/advanced/cpp_extension.html)

Build steps:

- Build C++/CUDA extensions by going into the `cpp/` folder and executing `python setup.py install --user`,
- JIT-compile C++/CUDA extensions by going into the `cpp/` folder and calling `python jit.py`, which will JIT-compile the extension and load it,
- Benchmark Python vs. C++ vs. CUDA by running `python benchmark.py {py, cpp}`,
- Run gradient checks on the code by running `python grad_check.py {py, cpp}`.
- Run output checks on the code by running `python check.py {forward, backward}`.
