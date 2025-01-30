#include "rl_mesh_controller/actor_critic_network.h"

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
    // Define the graph
    tensorflow::GraphDef graph_def;

    // Define the actor network
    tensorflow::NodeDef* input = graph_def.add_node();
    input->set_name("input");
    input->set_op("Placeholder");
    (*input->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    (*input->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(-1);
    (*input->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(12);

    tensorflow::NodeDef* actor_fc1_weights = graph_def.add_node();
    actor_fc1_weights->set_name("actor_fc1_weights");
    actor_fc1_weights->set_op("Const");
    (*actor_fc1_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    (*actor_fc1_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
    (*actor_fc1_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(12);
    (*actor_fc1_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);

    tensorflow::NodeDef* actor_fc1_bias_weights = graph_def.add_node();
    actor_fc1_bias_weights->set_name("actor_fc1_bias_weights");
    actor_fc1_bias_weights->set_op("Const");
    (*actor_fc1_bias_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    (*actor_fc1_bias_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
    (*actor_fc1_bias_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);

    tensorflow::NodeDef* actor_fc1 = graph_def.add_node();
    actor_fc1->set_name("actor_fc1");
    actor_fc1->set_op("MatMul");
    (*actor_fc1->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    actor_fc1->add_input("input");
    actor_fc1->add_input("actor_fc1_weights");

    tensorflow::NodeDef* actor_fc1_bias = graph_def.add_node();
    actor_fc1_bias->set_name("actor_fc1_bias");
    actor_fc1_bias->set_op("BiasAdd");
    (*actor_fc1_bias->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    actor_fc1_bias->add_input("actor_fc1");
    actor_fc1_bias->add_input("actor_fc1_bias_weights");

    tensorflow::NodeDef* actor_fc1_relu = graph_def.add_node();
    actor_fc1_relu->set_name("actor_fc1_relu");
    actor_fc1_relu->set_op("Relu");
    (*actor_fc1_relu->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    actor_fc1_relu->add_input("actor_fc1_bias");

    tensorflow::NodeDef* actor_fc2_weights = graph_def.add_node();
    actor_fc2_weights->set_name("actor_fc2_weights");
    actor_fc2_weights->set_op("Const");
    (*actor_fc2_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    (*actor_fc2_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
    (*actor_fc2_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);
    (*actor_fc2_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);

    tensorflow::NodeDef* actor_fc2_bias_weights = graph_def.add_node();
    actor_fc2_bias_weights->set_name("actor_fc2_bias_weights");
    actor_fc2_bias_weights->set_op("Const");
    (*actor_fc2_bias_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    (*actor_fc2_bias_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
    (*actor_fc2_bias_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);

    tensorflow::NodeDef* actor_fc2 = graph_def.add_node();
    actor_fc2->set_name("actor_fc2");
    actor_fc2->set_op("MatMul");
    (*actor_fc2->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    actor_fc2->add_input("actor_fc1_relu");
    actor_fc2->add_input("actor_fc2_weights");

    tensorflow::NodeDef* actor_fc2_bias = graph_def.add_node();
    actor_fc2_bias->set_name("actor_fc2_bias");
    actor_fc2_bias->set_op("BiasAdd");
    (*actor_fc2_bias->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    actor_fc2_bias->add_input("actor_fc2");
    actor_fc2_bias->add_input("actor_fc2_bias_weights");

    tensorflow::NodeDef* actor_fc2_relu = graph_def.add_node();
    actor_fc2_relu->set_name("actor_fc2_relu");
    actor_fc2_relu->set_op("Relu");
    (*actor_fc2_relu->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    actor_fc2_relu->add_input("actor_fc2_bias");

    tensorflow::NodeDef* actor_output_weights = graph_def.add_node();
    actor_output_weights->set_name("actor_output_weights");
    actor_output_weights->set_op("Const");
    (*actor_output_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    (*actor_output_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
    (*actor_output_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);
    (*actor_output_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(3);

    tensorflow::NodeDef* actor_output_bias_weights = graph_def.add_node();
    actor_output_bias_weights->set_name("actor_output_bias_weights");
    actor_output_bias_weights->set_op("Const");
    (*actor_output_bias_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    (*actor_output_bias_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
    (*actor_output_bias_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(3);

    tensorflow::NodeDef* actor_output = graph_def.add_node();
    actor_output->set_name("actor_output");
    actor_output->set_op("MatMul");
    (*actor_output->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    actor_output->add_input("actor_fc2_relu");
    actor_output->add_input("actor_output_weights");

    tensorflow::NodeDef* actor_output_bias = graph_def.add_node();
    actor_output_bias->set_name("actor_output_bias");
    actor_output_bias->set_op("BiasAdd");
    (*actor_output_bias->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    actor_output_bias->add_input("actor_output");
    actor_output_bias->add_input("actor_output_bias_weights");

    tensorflow::NodeDef* actor_output_tanh = graph_def.add_node();
    actor_output_tanh->set_name("actor_output_tanh");
    actor_output_tanh->set_op("Tanh");
    (*actor_output_tanh->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    actor_output_tanh->add_input("actor_output_bias");

    // Create the graph in the session
    tensorflow::Status status = session_->Create(graph_def);
    if (!status.ok()) {
        throw std::runtime_error("Failed to create graph: " + status.ToString());
    }
}

std::vector<tensorflow::Tensor> ActorCriticNetwork::Predict(const tensorflow::Tensor& input) {
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status status = session_->Run({{"input", input}}, {"actor_output_tanh"}, {}, &outputs);
    if (!status.ok()) {
        throw std::runtime_error("Failed to run session: " + status.ToString());
    }
    return outputs;
}