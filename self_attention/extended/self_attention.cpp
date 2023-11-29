#include <torch/extension.h>
#include <vector>
#include <iostream>

std::vector<torch::Tensor> self_attention_forward(
    std::vector<torch::Tensor> vkq,
    std::vector<torch::Tensor> weights,
    torch::optional<torch::Tensor> mask, std::vector<int64_t> consts)
{
    // Unpacking tensors and constants
    int64_t N = consts[0], seq_len = consts[1], heads = consts[2], head_dim = consts[3], embed_size = consts[4];
    torch::Tensor values = vkq[0], keys = vkq[1], query = vkq[2];
    torch::Tensor W_values = weights[0], W_keys = weights[1], W_query = weights[2], W_out = weights[3], B_out = weights[4];

    values = values.reshape({N, seq_len, heads, head_dim}).transpose(1, 2).reshape({N * heads, seq_len, head_dim});
    keys = keys.reshape({N, seq_len, heads, head_dim}).transpose(1, 2).reshape({N * heads, seq_len, head_dim});
    query = query.reshape({N, seq_len, heads, head_dim}).transpose(1, 2).reshape({N * heads, seq_len, head_dim});

    torch::Tensor values_mod = torch::matmul(values, W_values.transpose(0, 1));
    torch::Tensor keys_mod = torch::matmul(keys, W_keys.transpose(0, 1));
    torch::Tensor query_mod = torch::matmul(query, W_query.transpose(0, 1));

    return {
        out,
        scores,
        attention,
        values_mod,
        keys_mod,
        query_mod,
    };
}

torch::Tensor d_softmax(torch::Tensor output, torch::Tensor d_output)
{
    torch::Tensor ds = (output * d_output).sum();
    return output * (d_output - ds);
}

std::vector<torch::Tensor> self_attention_backward(
    torch::Tensor grad_out,
    torch::Tensor scores,
    torch::Tensor attention,
    torch::Tensor values_mod,
    torch::Tensor keys_mod,
    torch::Tensor query_mod,
    std::vector<torch::Tensor> vkq,
    std::vector<torch::Tensor> weights,
    std::vector<int64_t> consts)
{
    int log_ind = 0;

    // Unpacking tensors and constants
    int64_t N = consts[0], seq_len = consts[1], heads = consts[2], head_dim = consts[3], embed_size = consts[4];
    torch::Tensor values = vkq[0], keys = vkq[1], query = vkq[2];
    torch::Tensor W_values = weights[0], W_keys = weights[1], W_query = weights[2], W_out = weights[3], B_out = weights[4];

    torch::Tensor grad_B_out = grad_out.sum(std::vector<int64_t>({0, 1}));

    torch::Tensor scores_temp = scores.reshape({N, heads, seq_len, head_dim}).transpose(1, 2).reshape({N, seq_len, heads * head_dim});
    torch::Tensor grad_W_out = torch::matmul(grad_out.transpose(1, 2), scores_temp).sum(0);
    torch::Tensor grad_scores = torch::matmul(grad_out, W_out).reshape({N, seq_len, heads, head_dim}).transpose(1, 2).reshape({N * heads, seq_len, head_dim});

        torch::Tensor grad_W_values = torch::matmul(grad_values_mod.transpose(1, 2), values_temp).sum(0);
    torch::Tensor grad_W_keys = torch::matmul(grad_keys_mod.transpose(1, 2), keys_temp).sum(0);
    torch::Tensor grad_W_query = torch::matmul(grad_query_mod.transpose(1, 2), query_temp).sum(0);

    torch::Tensor grad_values = torch::matmul(grad_values_mod, W_values).reshape({N, heads, seq_len, head_dim}).transpose(1, 2).reshape({N, seq_len, heads * head_dim});
    torch::Tensor grad_keys = torch::matmul(grad_keys_mod, W_keys).reshape({N, heads, seq_len, head_dim}).transpose(1, 2).reshape({N, seq_len, heads * head_dim});
    torch::Tensor grad_query = torch::matmul(grad_query_mod, W_query).reshape({N, heads, seq_len, head_dim}).transpose(1, 2).reshape({N, seq_len, heads * head_dim});

    return {
        grad_values,
        grad_keys,
        grad_W_out,
        grad_B_out,
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &self_attention_forward, "SelfAttention forward");
    m.def("backward", &self_attention_backward, "SelfAttention backward");
}
