#include <iostream>
#include "tensor_load.hpp"
#include "train.hpp"



int main(int argc, char* argv[])
{
    Tensor arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    Tensor arr2
      {{5.0, 6.0, 7.0}};

    auto attention_layer = std::make_unique<Attention>(3, 3);
    auto linr = std::make_unique<Linear>(3,3);
    auto linr2 = std::make_unique<Linear>(3,3);
    auto linr3 = std::make_unique<Linear>(3,3);
    auto tanh_obj = std::make_unique<Tanh>();

    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::move(linr));
    layers.push_back(std::move(attention_layer));
    layers.push_back(std::move(linr2));
    layers.push_back(std::move(linr3));
    layers.push_back(std::move(tanh_obj));

    NeuralNet nn(std::move(layers));
    Tensor inputs = arr1;
    Tensor targets = arr2;
    int num_epochs = 10;
    BatchIterator batch_it(1,true);
    MSE mse;
    SGD optim(0.01);
    Adam optim2(0.01);

    Train train_obj;
    train_obj.train(nn,inputs,targets,num_epochs,batch_it,mse,optim2);


    return 0;
}