#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP


#include "tensor_load.hpp"
#include "layers.hpp"
#include "conv2d.hpp"
#include <memory>


class NeuralNet{
    public:
        std::vector<std::unique_ptr<Layer>> layers_class;
        NeuralNet(std::vector<std::unique_ptr<Layer>> layers):layers_class(std::move(layers)) {}

        Tensor forward(Tensor& inputs){
            for(auto& layer : layers_class)
                inputs = layer->forward(inputs);
            return inputs;
        }

        Tensor backward(Tensor& grad,Tensor inputs){
            Tensor res = grad;
            auto iter = layers_class.rbegin();
            auto end = layers_class.rend();
            while(iter != end)
            {
            res = (*iter)->backward(res,inputs);
            ++iter;
            }

            return res;
        }

        std::vector<TensorTuple> params_and_grads() {
            std::vector<TensorTuple> result;
            std::string name_weight = "weight";
            std::string name_bias = "bias";
            for (auto& layer : layers_class) {
                result.push_back(std::make_tuple(name_weight,layer->params.weights,layer->params.grad_weights));
                result.push_back(std::make_tuple(name_bias,layer->params.bias,layer->params.grad_biases));
            }
            return result;
        }
};


#endif       



    