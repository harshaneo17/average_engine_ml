#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "tensor_load.hpp"
#include "layers.hpp"
#include <xtensor/xview.hpp>

class Conv2D : public Layer {
public:
    Conv2D(int in_channels, int out_channels, std::vector<int> kernel_size, int stride = 1, int padding = 0)
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
        
        int batch_size = inputs.shape(0);
        int input_height = inputs.shape(2);
        int input_width = inputs.shape(3);
        
        int output_height = (input_height - kernel_size[0] + 2 * padding) / stride + 1;
        int output_width = (input_width - kernel_size[1] + 2 * padding) / stride + 1;
        std::cout << "train yikes" << std::endl;
        std::vector<std::size_t> shape = {
                                            static_cast<std::size_t>(batch_size),
                                            static_cast<std::size_t>(out_channels),
                                            static_cast<std::size_t>(output_height),
                                            static_cast<std::size_t>(output_width)
                                        };

        std::vector<double> data(batch_size * out_channels * output_height * output_width, 0.0);

        Tensor outputs(shape, data.begin(), data.end());
        std::cout << "debug 1st forward conv2d" << std::endl;
        // Implement convolution operation
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        double sum = 0.0;
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size[0]; ++kh) {
                                for (int kw = 0; kw < kernel_size[1]; ++kw) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
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
        int batch_size = inputs.shape(0);
        int input_height = inputs.shape(2);
        int input_width = inputs.shape(3);
        
        int output_height = grad.shape(2);
        int output_width = grad.shape(3);

        Tensor input_grad = xt::zeros<double>(inputs.shape());
        params.grad_weights = xt::zeros<double>(params.weights.shape());
        params.grad_biases = xt::sum(grad, {0, 2, 3});

        std::cout << "debug 1st backward conv2d" << std::endl;
        // Compute gradients
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size[0]; ++kh) {
                                for (int kw = 0; kw < kernel_size[1]; ++kw) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
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
    int in_channels;
    int out_channels;
    std::vector<int> kernel_size;
    int stride;
    int padding;
};

#endif