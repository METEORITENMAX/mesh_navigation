#ifndef RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H
#define RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H

#include <torch/torch.h>
#include <iostream>
#include <vector>

// Define a custom module
struct CustomModuleImpl : torch::nn::Module {
    CustomModuleImpl(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        // Register parameters
        W = register_parameter("W", torch::randn({input_size, hidden_size}));
        b = register_parameter("b", torch::randn(hidden_size));

        // Register submodules
        linear = register_module("linear", torch::nn::Linear(hidden_size, output_size));
    }

    // Implement the forward method
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::addmm(b, x, W));
        x = torch::tanh(linear->forward(x));
        return x;
    }

    torch::Tensor W, b;
    torch::nn::Linear linear{nullptr};
};

class ActorCriticNetwork {
public:
    ActorCriticNetwork();
    ~ActorCriticNetwork();

    void initializeGraph();
    void SaveModel();

private:
    std::shared_ptr<CustomModuleImpl> model;

};

#endif  // RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H