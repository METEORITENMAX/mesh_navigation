#include "rl_mesh_controller/actor_critic_network.h"
#include <rclcpp/rclcpp.hpp>

ActorCriticNetwork::ActorCriticNetwork() {
    RCLCPP_INFO(rclcpp::get_logger("ActorCriticNetwork"), "Initializing ActorCriticNetwork...");
    model = std::make_shared<CustomModuleImpl>(12, 256, 3);
}

ActorCriticNetwork::~ActorCriticNetwork() {
    RCLCPP_INFO(rclcpp::get_logger("ActorCriticNetwork"), "ActorCriticNetwork destroyed.");
}

void ActorCriticNetwork::initializeGraph() {
    torch::Tensor input = torch::randn({1, 12});
    std::ostringstream oss;
    oss << input;
    RCLCPP_INFO(rclcpp::get_logger("ActorCriticNetwork"), "Input tensor: %s", oss.str().c_str());

    try {
        c10::InferenceMode guard(true);  // Only enable if model is initialized
        torch::Tensor output = model->forward(input);

        oss.str("");
        oss.clear();
        oss << output;
        RCLCPP_INFO(rclcpp::get_logger("ActorCriticNetwork"), "Output tensor: %s", oss.str().c_str());
    } catch (const std::exception &e) {
        RCLCPP_ERROR(rclcpp::get_logger("ActorCriticNetwork"), "Error during forward pass: %s", e.what());
    }
}

void ActorCriticNetwork::SaveModel() {

}