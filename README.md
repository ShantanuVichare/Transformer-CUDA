# PyTorch C++/CUDA extension for SelfAttention Transformer layer

Install and Build steps:

- (Optional) Create Python environment using venv or conda.
- `cd` into `self_attention/` directory
- Setup required packages using `pip install -r requirements.txt`.
- Build extensions by going into the `extended/` folder and executing `python setup.py install --user`
- JIT-compile extensions by going into the `extended/` folder and calling `python jit.py`, which will JIT-compile the extension and load it
- Run tests and generate plots using `test.ipynb`
