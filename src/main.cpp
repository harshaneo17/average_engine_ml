#include <iostream>
#include "tensor_load.hpp"
#include "train.hpp"
#include "activations.hpp"
#include "layers.hpp"


int main() {
    // Expanded input data: 5 sentences, each with 10-dimensional embeddings
    Tensor input_data = {
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, // Sentence 1
        {0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7, 1.0, 0.9}, // Sentence 2
        {0.3, 0.4, 0.2, 0.5, 0.7, 0.6, 0.9, 1.0, 0.8, 0.1}, // Sentence 3
        {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3}, // Sentence 4
        {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4}  // Sentence 5
    };

    auto shape = input_data.shape();
    input_data = input_data.reshape({shape[0], 1, shape[1]});
    
    // Expanded target data: one-hot encoded labels for 3 classes
    Tensor target_data = {
        {1.0, 0.0, 0.0},  // Label for class 1
        {0.0, 1.0, 0.0},  // Label for class 2
        {0.0, 0.0, 1.0},  // Label for class 3
        {1.0, 0.0, 0.0},  // Label for class 1
        {0.0, 1.0, 0.0}   // Label for class 2
    };

    // Define the layers
    auto attention_layer = std::make_unique<Attention>(10, 64);  // Input size 10 (embedding dimension), hidden size 64
    auto activation1 = std::make_unique<Tanh>();
    auto output_layer = std::make_unique<Linear>(64, 3);         // Output size 3 (for 3 classes)
    auto activation2 = std::make_unique<Softmax>();

    // Add layers to the network
    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::move(attention_layer));
    layers.push_back(std::move(activation1));
    layers.push_back(std::move(output_layer));
    layers.push_back(std::move(activation2));

    // Create neural network
    NeuralNet nn(std::move(layers));

    // Training parameters
    int num_epochs = 20;
    BatchIterator batch_it(1, true);
    CrossEntropy ce_loss;  // Assuming you have a cross-entropy loss class
    Adam optim(0.001);

    // Train the network
    Train train_obj;
    train_obj.train(nn, input_data, target_data, num_epochs, batch_it, ce_loss, optim);

    return 0;
}
