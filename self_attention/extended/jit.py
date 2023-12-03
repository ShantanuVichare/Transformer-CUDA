from torch.utils.cpp_extension import load

self_attention = load(
    name="self_attention", sources=["self_attention.cpp"], verbose=True
)
help(self_attention)
