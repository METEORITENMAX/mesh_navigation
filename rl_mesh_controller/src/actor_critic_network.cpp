#include "rl_mesh_controller/actor_critic_network.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/training_ops.h"
#include "tensorflow/cc/framework/gradients.h"

ActorCriticNetwork::ActorCriticNetwork()  : root(tensorflow::Scope::NewRootScope()) {
    // Create a new TensorFlow session
    tensorflow::SessionOptions options;
    tensorflow::Status status = tensorflow::NewSession(options, &session_);
    if (!status.ok()) {
        throw std::runtime_error("Failed to create TensorFlow session: " + status.ToString());
    }
}

ActorCriticNetwork::~ActorCriticNetwork() {


    // Close the session
    tensorflow::Status status = session_->Close();
    if (!status.ok()) {
        std::cerr << "Failed to close TensorFlow session: " + status.ToString() << std::endl;
    }
}

void ActorCriticNetwork::initializeGraph() {
    try {
        // Define the computational graph
        root = tensorflow::Scope::NewRootScope();

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
        status = root.ToGraphDef(&graph_def_);
        if (!status.ok()) {
            throw std::runtime_error("Failed to create graph: " + status.ToString());
        }

        status = session_->Create(graph_def_);
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
    learning_rate.scalar<float>()() = 0.001;

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

void ActorCriticNetwork::SaveGraph(tensorflow::Session* session, const std::string& export_path) {
    tensorflow::GraphDef graph_def;
    tensorflow::Status status = root.ToGraphDef(&graph_def);
    if (!status.ok()) {
        std::cerr << "Error obtaining graph definition: " << status.ToString() << std::endl;
        return;
    }
    status = tensorflow::WriteTextProto(tensorflow::Env::Default(), export_path, graph_def);
    if (!status.ok()) {
        std::cerr << "Error saving graph: " << status.ToString() << std::endl;
    } else {
        std::cout << "Graph saved successfully to " << export_path << std::endl;
    }
}

void ActorCriticNetwork::SaveWeights(tensorflow::Session* session, const std::string& export_path) {
    std::vector<tensorflow::Tensor> output_tensors;
    tensorflow::Status status = session->Run({}, {"fc1_weights", "fc2_weights"}, {}, &output_tensors);
    if (!status.ok()) {
        std::cerr << "Error running session to get weights: " << status.ToString() << std::endl;
        return;
    }

    tensorflow::checkpoint::TensorSliceWriter writer(export_path, tensorflow::checkpoint::CreateTableTensorSliceBuilder);
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        const tensorflow::Tensor& tensor = output_tensors[i];
        tensorflow::TensorShape shape = tensor.shape();
        tensorflow::TensorSlice slice = tensorflow::TensorSlice::ParseOrDie("-:-");
        status = writer.Add("fc" + std::to_string(i + 1) + "_weights", shape, slice, tensor.flat<float>().data());
        if (!status.ok()) {
            std::cerr << "Error adding tensor to writer: " << status.ToString() << std::endl;
            return;
        }
    }

    status = writer.Finish();
    if (!status.ok()) {
        std::cerr << "Error finishing writer: " << status.ToString() << std::endl;
    } else {
        std::cout << "Weights saved successfully to " << export_path << std::endl;
    }
}

void ActorCriticNetwork::SaveModel() {
    // Verzeichnis für das Modell definieren
    std::string export_dir = "/tmp/tensorflow_model";  // Speicherort anpassen

    // Erstelle das Verzeichnis falls es nicht existiert
    struct stat info;
    if (stat(export_dir.c_str(), &info) != 0) {
        if (mkdir(export_dir.c_str(), 0777) == -1) {
            std::cerr << "Error creating directory: " << strerror(errno) << std::endl;
            return;
        }
    }

    // Checkpoint-Dateiname definieren
    std::string checkpoint_prefix = export_dir + "/model_checkpoint";

    // Tensor für den Speicherpfad erstellen
    tensorflow::Tensor checkpoint_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
    checkpoint_tensor.scalar<tensorflow::tstring>()() = checkpoint_prefix;

    // Speicheroperation korrekt definieren
    auto tensor_names = tensorflow::ops::Const(root.WithOpName("tensor_names"),
                                               {"fc1_weights", "fc2_weights"},
                                               tensorflow::TensorShape({2}));

    auto save_op = tensorflow::ops::Save(root.WithOpName("save_op"),
                                         checkpoint_tensor,
                                         tensor_names,
                                         {fc1_weights, fc2_weights});


    checkpoint_tensor.scalar<tensorflow::tstring>()() = "/tmp/tensorflow_model/model.ckpt";
    tensorflow::Status status;
    status = session_->Run(
        {{"save/Const", checkpoint_tensor}},
        {},
        {"save/control_dependency"},
        nullptr
    );


    if (!status.ok()) {
        std::cerr << "Error saving model: " << status.ToString() << std::endl;
    } else {
        std::cout << "Model saved successfully at " << checkpoint_prefix << std::endl;
    }
}