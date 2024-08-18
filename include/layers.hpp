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
            Tensor prod = inputs * params.weights;
            Tensor outputs = prod + params.bias;
            // std::cout << "These are outputs from forward" << outputs << std::endl;
            return outputs;
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

};


#endif

