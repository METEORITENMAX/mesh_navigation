#ifndef RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H
#define RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/platform/env.h>
#include <iostream>
#include <vector>

class ActorCriticNetwork {
public:
    ActorCriticNetwork();
    ~ActorCriticNetwork();

    void initializeGraph();

    std::vector<tensorflow::Tensor> Predict(const tensorflow::Tensor& input);
    void Train(const tensorflow::Tensor& states, const tensorflow::Tensor& actions, const tensorflow::Tensor& rewards, const tensorflow::Tensor& next_states);


private:
    tensorflow::Session* session_;
};

#endif  // RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H