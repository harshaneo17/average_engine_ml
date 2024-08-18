#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include "tensor_load.hpp"
#include "layers.hpp"
#include "activations.hpp"

struct AttentionParams {
    Tensor weights_query;
    Tensor weights_key;
    Tensor weights_value;
    Tensor bias_query;
    Tensor bias_key;
    Tensor bias_value;
    Tensor grad_weights_query;
    Tensor grad_weights_key;
    Tensor grad_weights_value;
    Tensor grad_bias_query;
    Tensor grad_bias_key;
    Tensor grad_bias_value;
};

class Attention : public Layer {
public:
    Attention(double input_size, double hidden_size)
        : input_size(input_size), hidden_size(hidden_size) {initialize();}

    void initialize() {
        params.weights_query = xt::random::randn<double>({input_size, hidden_size});
        params.weights_key = xt::random::randn<double>({input_size, hidden_size});
        params.weights_value = xt::random::randn<double>({input_size, hidden_size});
        params.bias_query = xt::random::randn<double>({hidden_size});
        params.bias_key = xt::random::randn<double>({hidden_size});
        params.bias_value = xt::random::randn<double>({hidden_size});
    }

    Tensor forward(Tensor inputs) override {

        // inputs shape: (batch_size, seq_len, input_size)
        auto batch_size = inputs.shape()[0];
        auto seq_len = inputs.shape()[1];

        Tensor queries = dot(inputs, params.weights_query);
        queries = queries + xt::view(params.bias_query, xt::newaxis(), xt::all());
        
        Tensor keys = dot(inputs, params.weights_key);
        keys = keys + xt::view(params.bias_key, xt::newaxis(), xt::all());
        
        Tensor values = dot(inputs, params.weights_value);
        values = values + xt::view(params.bias_value, xt::newaxis(), xt::all());

        // Transpose keys for batch matrix multiplication
        auto keys_t = xt::transpose(keys, {0, 2, 1});

        Tensor scores = dot(queries, keys_t) / std::sqrt(hidden_size);

        Softmax softmax;
        Tensor attention_weights = softmax.forward(scores);

        Tensor output = dot(attention_weights, values);

        // Flatten the output
        output = output.reshape({batch_size, static_cast<size_t>(seq_len * hidden_size)});

        this->queries = queries;
        this->keys = keys;
        this->values = values;
        this->attention_weights = attention_weights;

        return output;
    }

    Tensor backward(Tensor grad, Tensor inputs) override {
        Tensor grad_values = dot(xt::transpose(attention_weights), grad);
        Tensor grad_attention_weights = dot(grad, xt::transpose(values));

        Softmax softmax;
        Tensor grad_scores = grad_attention_weights * softmax.backward(grad_attention_weights, attention_weights);

        Tensor grad_queries = dot(grad_scores, keys);
        Tensor grad_keys = dot(xt::transpose(grad_scores), queries);

        params.grad_weights_query = dot(xt::transpose(inputs), grad_queries);
        params.grad_weights_key = dot(xt::transpose(inputs), grad_keys);
        params.grad_weights_value = dot(xt::transpose(inputs), grad_values);

        params.grad_bias_query = xt::sum(grad_queries, {0});
        params.grad_bias_key = xt::sum(grad_keys, {0});
        params.grad_bias_value = xt::sum(grad_values, {0});

        Tensor grad_inputs = dot(grad_queries, xt::transpose(params.weights_query)) +
                             dot(grad_keys, xt::transpose(params.weights_key)) +
                             dot(grad_values, xt::transpose(params.weights_value));

        return grad_inputs;
    }

private:
    double input_size;
    double hidden_size;
    Tensor queries, keys, values, attention_weights;
    AttentionParams params;

    Tensor dot(const Tensor& a, const Tensor& b) {
    auto a_dims = a.dimension();
    auto b_dims = b.dimension();
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    if (a_dims == 2 && b_dims == 2) {
        // 2D x 2D
        if (a_shape[1] != b_shape[0]) {
            throw std::invalid_argument("Shapes of input tensors are incompatible for dot product");
        }

        Tensor result = xt::zeros<double>({a_shape[0], b_shape[1]});

        for (size_t i = 0; i < a_shape[0]; ++i) {
            for (size_t j = 0; j < b_shape[1]; ++j) {
                for (size_t k = 0; k < a_shape[1]; ++k) {
                    result(i, j) += a(i, k) * b(k, j);
                }
            }
        }

        return result;
    } else if (a_dims == 3 && b_dims == 3) {
        // 3D x 3D (batch matrix multiplication)
        if (a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1]) {
            throw std::invalid_argument("Shapes of input tensors are incompatible for dot product");
        }

        Tensor result = xt::zeros<double>({a_shape[0], a_shape[1], b_shape[2]});

        for (size_t n = 0; n < a_shape[0]; ++n) {
            for (size_t i = 0; i < a_shape[1]; ++i) {
                for (size_t j = 0; j < b_shape[2]; ++j) {
                    for (size_t k = 0; k < a_shape[2]; ++k) {
                        result(n, i, j) += a(n, i, k) * b(n, k, j);
                    }
                }
            }
        }

        return result;
    } else if (a_dims == 3 && b_dims == 2) {
        // 3D x 2D (special case for attention weights)
        if (a_shape[2] != b_shape[0]) {
            throw std::invalid_argument("Shapes of input tensors are incompatible for dot product");
        }

        Tensor result = xt::zeros<double>({a_shape[0], a_shape[1], b_shape[1]});

        for (size_t n = 0; n < a_shape[0]; ++n) {
            for (size_t i = 0; i < a_shape[1]; ++i) {
                for (size_t j = 0; j < b_shape[1]; ++j) {
                    for (size_t k = 0; k < a_shape[2]; ++k) {
                        result(n, i, j) += a(n, i, k) * b(k, j);
                    }
                }
            }
        }

        return result;
    } else {
        throw std::invalid_argument("Dot product requires 2D or 3D tensors");
    }
}

};

#endif
