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
        : input_size(input_size), hidden_size(hidden_size) {}

    void initialize() {
        params.weights_query = xt::random::randn<double>({input_size, hidden_size});
        params.weights_key = xt::random::randn<double>({input_size, hidden_size});
        params.weights_value = xt::random::randn<double>({input_size, hidden_size});
        params.bias_query = xt::random::randn<double>({hidden_size});
        params.bias_key = xt::random::randn<double>({hidden_size});
        params.bias_value = xt::random::randn<double>({hidden_size});
    }

    Tensor forward(Tensor inputs) override {
        initialize();

        Tensor queries = dot(inputs, params.weights_query) + params.bias_query;
        Tensor keys = dot(inputs, params.weights_key) + params.bias_key;
        Tensor values = dot(inputs, params.weights_value) + params.bias_value;

        Tensor scores = dot(queries, xt::transpose(keys)) / std::sqrt(hidden_size);

        Softmax softmax;
        Tensor attention_weights = softmax.forward(scores);

        Tensor output = dot(attention_weights, values);

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
    if (a.dimension() != 2 || b.dimension() != 2) {
        throw std::invalid_argument("Dot product requires 2D tensors");
    }

    auto a_shape = a.shape();
    auto b_shape = b.shape();

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
}

};

#endif
