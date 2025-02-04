#ifndef RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H
#define RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/platform/env.h>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/util/tensor_slice_writer.h"
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <iostream>
#include <cstring>
#include <cerrno>

class ActorCriticNetwork {
public:
    ActorCriticNetwork();
    ~ActorCriticNetwork();

    void initializeGraph();

    std::vector<tensorflow::Tensor> Predict(const tensorflow::Tensor& input);
    void Train(const tensorflow::Tensor& states, const tensorflow::Tensor& actions, const tensorflow::Tensor& rewards, const tensorflow::Tensor& next_states);

    void SaveModel();
    void SaveGraph(tensorflow::Session* session, const std::string& export_path);
    void SaveWeights(tensorflow::Session* session, const std::string& export_path);
private:
    tensorflow::Session* session_;
    tensorflow::GraphDef graph_def_;
    tensorflow::Scope root;
    tensorflow::Output fc1_weights;
    tensorflow::Output fc2_weights;

};

#endif  // RL_MESH_CONTROLLER__ACTOR_CRITIC_NETWORK_H
