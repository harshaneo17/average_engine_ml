#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"

struct Params{
  Tensor weights;
  Tensor bias;
  Tensor grad_weights;
  Tensor grad_biases;
};

struct Layer {
    public:
        virtual Tensor forward(Tensor inputs) {}
        virtual Tensor backward(Tensor grad, Tensor inputs){}
        Params params;
};

class Linear : public Layer {
    public:
        /*computes output = inputs @ weights + biases*/
        Linear (double input_size,double output_size) : input_class_size(input_size),output_class_size(output_size){initialize();}
        Tensor weights,bias,grad_weights,grad_bias;
        Params params;

        void initialize(){
            params.weights = xt::random::randn<double>({input_class_size,output_class_size});
            params.bias = xt::random::randn<double>({output_class_size});
        }
        
        Tensor forward(Tensor inputs) override {
            /*outputs = inputs @ w + b*/
            /*Mathematically, a linear layer can be represented as:
                Y = XW + b
                where:
                X is the input vector of size n x m, where n is the batch size and m is the number of input features.
                W is the weight matrix of size m x p, where p is the number of output features.
                b is the bias vector of size p.
                Y is the output vector of size n x p*/
            auto batch_size = inputs.shape()[0];
            auto seq_len = inputs.shape()[1];
            inputs = inputs.reshape({batch_size * seq_len, static_cast<unsigned long>(input_class_size)});
            
            Tensor output = dot(inputs, weights) + bias;
            output = output.reshape({batch_size, seq_len, static_cast<unsigned long>(output_class_size)});
            return output;

        }

        Tensor backward(Tensor grad, Tensor inputs) override {
            /*
            if y = f(x) and x = a * b + c
            then dy/da = f'(x) * b
            and dy/db = f'(x) * a
            and dy/dc = f'(x)

            if y = f(x) and x = a @ b + c
            then dy/da = f'(x) @ b.T
            and dy/db = a.T @ f'(x)
            and dy/dc = f'(x)*/
            Tensor copy_var = xt::sum(grad,1);
            params.grad_biases = copy_var;
            Tensor tr_inputs = xt::transpose(inputs);
            params.grad_weights = tr_inputs * grad;
            auto tr_grad_w = xt::transpose(params.grad_weights);
            Tensor backward_outputs = grad * tr_grad_w;
            // std::cout << "These are outputs from backward" << backward_outputs << std::endl;
            return backward_outputs;
        }
        private:
            double input_class_size;
            double output_class_size;

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

