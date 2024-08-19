#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "tensor_load.hpp"
#include "layers.hpp"

class Tanh : public Layer {
public:
    Tensor forward(Tensor x) override {
        return tanh(x);
    }

    Tensor backward(Tensor grad,Tensor inputs) override {
        auto y = tanh(inputs);
        return 1 - pow(y, 2);
    }

private:
    Tensor tanh(Tensor& x) {
        return xt::tanh(x);
    }
};

class Sigmoid : public Layer {
public:
    Tensor forward(Tensor x) override {
        return sigmoid(x);
    }

    Tensor backward(Tensor grad,Tensor inputs) override {
        auto y = sigmoid(inputs);
        return y * (1 - y);
    }

private:
    Tensor sigmoid(Tensor& x) {
        auto denom_sigmoid = 1 + xt::exp(x);
        return 1 / denom_sigmoid;
    }
};

class Relu : public Layer {
public:
    Tensor forward(Tensor inputs) override {
        return relu(inputs);
    }

    Tensor backward(Tensor grad,Tensor inputs) override {
        return xt::where(inputs > 0, 1, 0); // derivative of relu
    }

private:
    Tensor relu(Tensor& x) {
        return xt::maximum(x, 0);
    }
};

class Softmax : public Layer {
public:
    Tensor forward(Tensor x) override {
        auto shape = x.shape();
        auto max_vals = xt::amax(x, {1});
        
        // Convert max_vals to a regular tensor and then reshape
        Tensor max_vals_tensor = max_vals;
        std::vector<size_t> new_shape(shape.size(), 1);
        new_shape[0] = shape[0];
        auto max_vals_reshaped = max_vals_tensor.reshape(new_shape);
        
        auto exp_vals = xt::exp(x - max_vals_reshaped);
        auto sum_exp = xt::sum(exp_vals, {1});
        
        // Convert sum_exp to a regular tensor and then reshape
        Tensor sum_exp_tensor = sum_exp;
        auto sum_exp_reshaped = sum_exp_tensor.reshape(new_shape);
        
        return exp_vals / sum_exp_reshaped;
    }

    Tensor backward(Tensor grad, Tensor inputs) override {
        auto softmax_output = forward(inputs);
        auto shape = softmax_output.shape();
        
        // Compute sum(grad * softmax_output) along axis 1
        auto sum_grad_softmax = xt::sum(grad * softmax_output, {1});
        
        // Convert sum_grad_softmax to a regular tensor and then reshape
        Tensor sum_grad_softmax_tensor = sum_grad_softmax;
        std::vector<size_t> new_shape(shape.size(), 1);
        new_shape[0] = shape[0];
        auto sum_grad_softmax_reshaped = sum_grad_softmax_tensor.reshape(new_shape);
        
        return softmax_output * (grad - sum_grad_softmax_reshaped);
    }
};

#endif
