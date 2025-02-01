#include "rl_mesh_controller/actor_critic_network.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/training_ops.h"
#include "tensorflow/cc/framework/gradients.h"

ActorCriticNetwork::ActorCriticNetwork() {
    // Create a new TensorFlow session
    tensorflow::SessionOptions options;
    tensorflow::Status status = tensorflow::NewSession(options, &session_);
    if (!status.ok()) {
        throw std::runtime_error("Failed to create TensorFlow session: " + status.ToString());
    }
}

ActorCriticNetwork::~ActorCriticNetwork() {
    session_->Close();
}

void ActorCriticNetwork::initializeGraph() {
    try {
        // Define the computational graph
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();

        // Define placeholders
        auto states = tensorflow::ops::Placeholder(root.WithOpName("states"), tensorflow::DT_FLOAT);
        auto actions = tensorflow::ops::Placeholder(root.WithOpName("actions"), tensorflow::DT_FLOAT);
        auto rewards = tensorflow::ops::Placeholder(root.WithOpName("rewards"), tensorflow::DT_FLOAT);
        auto next_states = tensorflow::ops::Placeholder(root.WithOpName("next_states"), tensorflow::DT_FLOAT);

        // Define actor network layers (simplified)
        auto fc1_weights = tensorflow::ops::Variable(root.WithOpName("fc1_weights"), {12, 256}, tensorflow::DT_FLOAT);
        auto fc1_init = tensorflow::ops::Assign(root.WithOpName("fc1_init"), fc1_weights, tensorflow::ops::RandomNormal(root, {12, 256}, tensorflow::DT_FLOAT));
        auto fc1 = tensorflow::ops::MatMul(root.WithOpName("fc1"), states, fc1_weights);
        auto fc1_relu = tensorflow::ops::Relu(root.WithOpName("fc1_relu"), fc1);

        auto fc2_weights = tensorflow::ops::Variable(root.WithOpName("fc2_weights"), {256, 3}, tensorflow::DT_FLOAT);
        auto fc2_init = tensorflow::ops::Assign(root.WithOpName("fc2_init"), fc2_weights, tensorflow::ops::RandomNormal(root, {256, 3}, tensorflow::DT_FLOAT));
        auto fc2 = tensorflow::ops::MatMul(root.WithOpName("fc2"), fc1_relu, fc2_weights);
        auto actor_output = tensorflow::ops::Tanh(root.WithOpName("actor_output"), fc2);

        // Define loss function
        auto squared_diff = tensorflow::ops::SquaredDifference(root.WithOpName("squared_diff"), actions, actor_output);
        auto actor_loss = tensorflow::ops::Mean(root.WithOpName("actor_loss"), squared_diff, {0});

        // Define optimizer manually
        auto learning_rate = tensorflow::ops::Const(root.WithOpName("learning_rate"), 0.01f, {});
        std::vector<tensorflow::Output> grad_outputs;
        tensorflow::Status status = tensorflow::AddSymbolicGradients(root, {actor_loss}, {fc1_weights, fc2_weights}, &grad_outputs);
        if (!status.ok()) {
            throw std::runtime_error("Failed to add symbolic gradients: " + status.ToString());
        }
        auto apply_gradients_fc1 = tensorflow::ops::ApplyGradientDescent(root.WithOpName("apply_gradients_fc1"), fc1_weights, learning_rate, grad_outputs[0]);
        auto apply_gradients_fc2 = tensorflow::ops::ApplyGradientDescent(root.WithOpName("apply_gradients_fc2"), fc2_weights, learning_rate, grad_outputs[1]);

        // Create session
        tensorflow::GraphDef graph_def;
        status = root.ToGraphDef(&graph_def);
        if (!status.ok()) {
            throw std::runtime_error("Failed to create graph: " + status.ToString());
        }

        status = session_->Create(graph_def);
        if (!status.ok()) {
            throw std::runtime_error("Failed to create session: " + status.ToString());
        }

        // Run variable initialization
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
            {"states", tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 12}))},
            {"actions", tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 3}))},
            {"rewards", tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1}))},
            {"next_states", tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 12}))}
        };
        status = session_->Run(feed_dict, {}, {"fc1_init", "fc2_init"}, nullptr);
        if (!status.ok()) {
            throw std::runtime_error("Failed to initialize variables: " + status.ToString());
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error during graph initialization: " << e.what() << std::endl;
    }
}
std::vector<tensorflow::Tensor> ActorCriticNetwork::Predict(const tensorflow::Tensor& input) {
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status status = session_->Run({{"states", input}}, {"actor_output"}, {}, &outputs);
    if (!status.ok()) {
        throw std::runtime_error("Failed to run session: " + status.ToString());
    }
    return outputs;
}

void ActorCriticNetwork::Train(const tensorflow::Tensor& states, const tensorflow::Tensor& actions, const tensorflow::Tensor& rewards, const tensorflow::Tensor& next_states) {
    // Ensure all inputs are provided to the session
    std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
        {"states", states},
        {"actions", actions},
        {"rewards", rewards},
        {"next_states", next_states}
    };

    std::vector<tensorflow::Tensor> actor_loss;
    tensorflow::Status status = session_->Run(
        feed_dict,
        {"actor_loss"},
        {},
        &actor_loss
    );

    if (!status.ok()) {
        throw std::runtime_error("Failed to compute actor loss: " + status.ToString());
    }

    tensorflow::Tensor learning_rate(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
    learning_rate.scalar<float>()() = 0.01;

    feed_dict.push_back({"learning_rate", learning_rate});
    feed_dict.push_back({"actor_loss", actor_loss[0]});

    status = session_->Run(
        feed_dict,
        {},
        {"apply_gradients_fc1", "apply_gradients_fc2"},
        nullptr
    );

    if (!status.ok()) {
        throw std::runtime_error("Failed to train actor network: " + status.ToString());
    }
}