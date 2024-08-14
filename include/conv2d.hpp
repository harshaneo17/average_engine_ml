#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "tensor_load.hpp"
#include "layers.hpp"
#include <xtensor/xview.hpp>

class Conv2D : public Layer {
public:
    Conv2D(size_t in_channels, size_t out_channels, std::vector<size_t> kernel_size, size_t stride = 1, size_t padding = 0)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), 
        stride(stride), padding(padding) {
        initialize();
    }

    void initialize() {
        // Initialize weights and bias
        params.weights = xt::random::randn<double>({out_channels, in_channels, kernel_size[0], kernel_size[1]});
        params.bias = xt::random::randn<double>({out_channels});
    }

    Tensor forward(Tensor inputs) override {
    // Use size_t for dimensions
    size_t batch_size = inputs.shape(0);
    size_t input_channels = inputs.shape(1);
    size_t input_height = inputs.shape(2);
    size_t input_width = inputs.shape(3);

    std::cout << "Input shape: " << batch_size << "x" << input_channels << "x" << input_height << "x" << input_width << std::endl;

    // Check if stride is not zero
    if (stride == 0) {
    throw std::runtime_error("Stride cannot be zero");
    }

    // Use integer division for output dimensions
    int output_height = (static_cast<int>(input_height) - kernel_size[0] + 2 * padding) / stride + 1;
    int output_width = (static_cast<int>(input_width) - kernel_size[1] + 2 * padding) / stride + 1;

    std::cout << "Calculated output height: " << output_height << std::endl;
    std::cout << "Calculated output width: " << output_width << std::endl;

    // Check if output dimensions are positive
    if (output_height <= 0 || output_width <= 0) {
    throw std::runtime_error("Invalid output dimensions");
    }

    Tensor outputs;
    try {
        outputs = xt::zeros<double>({batch_size, 
                                     static_cast<size_t>(out_channels), 
                                     static_cast<size_t>(output_height), 
                                     static_cast<size_t>(output_width)});
        std::cout << "Output tensor created successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error creating output tensor: " << e.what() << std::endl;
        throw;
    }
    
    // Implement convolution operation
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < output_height; ++oh) {
                for (size_t ow = 0; ow < output_width; ++ow) {
                    double sum = 0.0;
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_size[0]; ++kh) {
                            for (size_t kw = 0; kw < kernel_size[1]; ++kw) {
                                size_t ih = oh * stride + kh - padding;
                                size_t iw = ow * stride + kw - padding;
                                if (ih < input_height && iw < input_width) {
                                    sum += inputs(b, ic, ih, iw) * params.weights(oc, ic, kh, kw);
                                }
                            }
                        }
                    }
                    outputs(b, oc, oh, ow) = sum + params.bias(oc);
                }
            }
        }
    }

    return outputs;
}

    Tensor backward(Tensor grad, Tensor inputs) override {
        double batch_size = inputs.shape(0);
        double input_height = inputs.shape(2);
        double input_width = inputs.shape(3);
        
        double output_height = grad.shape(2);
        double output_width = grad.shape(3);

        Tensor input_grad = xt::zeros<double>(inputs.shape());
        params.grad_weights = xt::zeros<double>(params.weights.shape());
        params.grad_biases = xt::sum(grad, {0, 2, 3});

        std::cout << "debug 1st backward conv2d" << std::endl;
        // Compute gradients
        for (double b = 0; b < batch_size; ++b) {
            for (double oc = 0; oc < out_channels; ++oc) {
                for (double oh = 0; oh < output_height; ++oh) {
                    for (double ow = 0; ow < output_width; ++ow) {
                        for (double ic = 0; ic < in_channels; ++ic) {
                            for (double kh = 0; kh < kernel_size[0]; ++kh) {
                                for (double kw = 0; kw < kernel_size[1]; ++kw) {
                                    double ih = oh * stride + kh - padding;
                                    double iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                        double grad_val = grad(b, oc, oh, ow);
                                        input_grad(b, ic, ih, iw) += grad_val * params.weights(oc, ic, kh, kw);
                                        params.grad_weights(oc, ic, kh, kw) += grad_val * inputs(b, ic, ih, iw);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return input_grad;
    }

private:
    size_t in_channels;
    size_t out_channels;
    std::vector<size_t> kernel_size;
    size_t stride;
    size_t padding;
};

#endif