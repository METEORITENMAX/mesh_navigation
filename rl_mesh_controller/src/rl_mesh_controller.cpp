/*
 *  Copyright 2020, Sebastian Pütz, Sabrina Frohn
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *  1. Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  authors:
 *    Sebastian Pütz <spuetz@uos.de>
 *
 */

#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/util/Meap.hpp>
#include <mbf_msgs/action/exe_path.hpp>
#include <rl_mesh_controller/rl_mesh_controller.h>
#include <mesh_map/util.h>
#include <pluginlib/class_list_macros.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <mbf_utility/exe_path_exception.h>

PLUGINLIB_EXPORT_CLASS(rl_mesh_controller::RLMeshController, mbf_mesh_core::MeshController);

#define DEBUG

#ifdef DEBUG
#define DEBUG_CALL(method) method
#else
#define DEBUG_CALL(method)
#endif

namespace rl_mesh_controller
{
RLMeshController::RLMeshController()
{

}

RLMeshController::~RLMeshController()
{
}

uint32_t RLMeshController::computeVelocityCommands(const geometry_msgs::msg::PoseStamped& pose,
                                                   const geometry_msgs::msg::TwistStamped& velocity,
                                                   geometry_msgs::msg::TwistStamped& cmd_vel,
                                                   std::string& message)
{
  const auto& mesh = map_ptr_->mesh();

  robot_pos_ = poseToPositionVector(pose);
  robot_dir_ = poseToDirectionVector(pose);
  std::array<float, 3> bary_coords;
  std::array<mesh_map::Vector, 3> vertices;

  if (!current_face_)
  {
    // initially search current face on complete map
    if (auto search_res_opt = map_ptr_->searchContainingFace(
            robot_pos_, config_.max_search_distance))
    {
      auto search_res = *search_res_opt;
      current_face_ = std::get<0>(search_res);
      vertices = std::get<1>(search_res);
      bary_coords = std::get<2>(search_res);

      // project position onto surface
      robot_pos_ = mesh_map::linearCombineBarycentricCoords(vertices, bary_coords);
    }
    else
    {
      // no corresponding face has been found
      return mbf_msgs::action::ExePath::Result::OUT_OF_MAP;
    }
  }
  else // current face is set
  {
    lvr2::FaceHandle face = current_face_.unwrap();
    vertices = mesh.getVertexPositionsOfFace(face);
    DEBUG_CALL(map_ptr_->publishDebugFace(face, mesh_map::color(1, 1, 1), "current_face");)
    DEBUG_CALL(map_ptr_->publishDebugPoint(robot_pos_, mesh_map::color(1, 1, 1), "robot_position");)

    float dist_to_surface;
    // check whether or not the position matches the current face
    // if not search for new current face
    if (mesh_map::projectedBarycentricCoords(
        robot_pos_, vertices, bary_coords, dist_to_surface)
        && dist_to_surface < config_.max_search_distance)
    {
      // current position is located inside and close enough to the face
      DEBUG_CALL(map_ptr_->publishDebugPoint(robot_pos_, mesh_map::color(0, 0, 1), "current_position");)
    }
    else if (auto search_res_opt = map_ptr_->searchNeighbourFaces(
                 robot_pos_, face, config_.max_search_radius, config_.max_search_distance))
    {
      // new face has been found out of the neighbour faces of the current face
      // update variables to new face
      auto search_res = *search_res_opt;
      current_face_ = face = std::get<0>(search_res);
      vertices = std::get<1>(search_res);
      bary_coords = std::get<2>(search_res);
      robot_pos_ = mesh_map::linearCombineBarycentricCoords(vertices, bary_coords);
      DEBUG_CALL(map_ptr_->publishDebugFace(face, mesh_map::color(1, 0.5, 0), "search_neighbour_face");)
      DEBUG_CALL(map_ptr_->publishDebugPoint(robot_pos_, mesh_map::color(0, 0, 1), "search_neighbour_pos");)
    }
    else if(auto search_res_opt = map_ptr_->searchContainingFace(
        robot_pos_, config_.max_search_distance))
    {
      // update variables to new face
      auto search_res = *search_res_opt;
      current_face_ = face = std::get<0>(search_res);
      vertices = std::get<1>(search_res);
      bary_coords = std::get<2>(search_res);
      robot_pos_ = mesh_map::linearCombineBarycentricCoords(vertices, bary_coords);
    }
    else
    {
      // no corresponding face has been found
      return mbf_msgs::action::ExePath::Result::OUT_OF_MAP;
    }
  }

  const lvr2::FaceHandle& face = current_face_.unwrap();
  std::array<lvr2::VertexHandle, 3> handles = map_ptr_->mesh_ptr->getVerticesOfFace(face);

  // update to which position of the plan the robot is closest

  const auto& opt_dir = map_ptr_->directionAtPosition(vector_map_, handles, bary_coords);
  if (!opt_dir)
  {
    DEBUG_CALL(map_ptr_->publishDebugFace(face, mesh_map::color(0.3, 0.4, 0), "no_directions");)
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Could not access vector field for the given face!");
    return mbf_msgs::action::ExePath::Result::FAILURE;
  }
  mesh_map::Normal mesh_dir = opt_dir.get().normalized();
  float cost = map_ptr_->costAtPosition(handles, bary_coords);
  const mesh_map::Normal& mesh_normal = poseToDirectionVector(pose, tf2::Vector3(0,0,1));
  std::array<float, 2> velocities = naiveControl(robot_pos_, robot_dir_, mesh_dir, mesh_normal, cost);

  // 1. Extract current state
  std::vector<float> state = get_state();

  // 2. Select action (inference or exploration)
  std::vector<float> action(3);
  // if (training_mode_ && rand() % 100 < 20) {  // 20% exploration
  //   action[0] = ((float)rand() / RAND_MAX) * config_.max_lin_velocity;
  //   action[1] = ((float)rand() / RAND_MAX) * config_.max_lin_velocity;
  //   action[2] = ((float)rand() / RAND_MAX) * config_.max_ang_velocity;
  // } else {
  //   torch::Tensor state_tensor = torch::from_blob(state.data(), {1, (long)state.size()});
  //   torch::Tensor action_tensor = actor_->forward(state_tensor);
  //   action[0] = action_tensor[0][0].item<float>() * config_.max_lin_velocity;
  //   action[1] = action_tensor[0][1].item<float>() * config_.max_lin_velocity;
  //   action[2] = action_tensor[0][2].item<float>() * config_.max_ang_velocity;
  // }


  cmd_vel.twist.linear.x = std::min(config_.max_lin_velocity, velocities[0] * config_.lin_vel_factor);
  cmd_vel.twist.angular.z = std::min(config_.max_ang_velocity, velocities[1] * config_.ang_vel_factor);
  // cmd_vel.twist.linear.x = std::min(static_cast<float>(config_.max_lin_velocity), action[0]);
  // cmd_vel.twist.linear.y = std::min(static_cast<float>(config_.max_lin_velocity), action[1]);
  // cmd_vel.twist.angular.z = std::min(static_cast<float>(config_.max_ang_velocity), action[2]);
  cmd_vel.header.stamp = node_->now();

  // 3. Execute action, get reward, and observe next state
  // Simulate reward calculation and next state update here...
  // float reward = -1.0f;  // Define reward function properly.
  // std::vector<float> next_state = get_state();  // Update based on action.

  // // 4. Store transition in replay buffer
  // if (replay_buffer_.size() >= replay_buffer_size_) {
  //   replay_buffer_.erase(replay_buffer_.begin());
  // }
  // replay_buffer_.emplace_back(state, action, reward, next_state);

  // // 5. Train model if in training mode
  // if (training_mode_ && replay_buffer_.size() > batch_size_) {
  //   trainModel();
  // }

  if (cancel_requested_)
  {
    return mbf_msgs::action::ExePath::Result::CANCELED;
  }
  return mbf_msgs::action::ExePath::Result::SUCCESS;
}

void RLMeshController::trainModel()
{
  RCLCPP_INFO(node_->get_logger(), "Training model...");
  // std::vector<size_t> batch_indices;
  // while (batch_indices.size() < batch_size_) {
  //   batch_indices.push_back(rand() % replay_buffer_.size());
  // }

  // for (size_t index : batch_indices) {
  //   auto [state, action, reward, next_state] = replay_buffer_[index];

  //   // Q-Learning Update
  //   std::vector<float> q_values = model_.Predict(state);
  //   std::vector<float> q_next_values = model_.Predict(next_state);
  //   float max_q_next = *std::max_element(q_next_values.begin(), q_next_values.end());
  //   q_values[/* action index */] = reward + gamma_ * max_q_next;

  //   // Train the model
  //   model_.Train(state, q_values, learning_rate_);
  // }
}


bool RLMeshController::isGoalReached(double dist_tolerance, double angle_tolerance)
{
  float goal_distance = (goal_pos_ - robot_pos_).length();
  float angle = acos(goal_dir_.dot(robot_dir_));
  return goal_distance <= static_cast<float>(dist_tolerance) && angle <= static_cast<float>(angle_tolerance);
}

bool RLMeshController::setPlan(const std::vector<geometry_msgs::msg::PoseStamped>& plan)
{
  // copy vector field // TODO just use vector field without copying

  RCLCPP_INFO(node_->get_logger(), "Received plan with %zu poses", plan.size());
  for (size_t i = 0; i < plan.size(); ++i)
  {
    const auto& pose = plan[i];
    RCLCPP_INFO(node_->get_logger(), "Pose %zu: Position (x: %f, y: %f, z: %f), Orientation (x: %f, y: %f, z: %f, w: %f)",
                i,
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
                pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w);
  }
  vector_map_ = map_ptr_->getVectorMap();
  DEBUG_CALL(map_ptr_->publishDebugPoint(poseToPositionVector(plan.front()), mesh_map::color(0, 1, 0), "plan_start");)
  DEBUG_CALL(map_ptr_->publishDebugPoint(poseToPositionVector(plan.back()), mesh_map::color(1, 0, 0), "plan_goal");)
  current_plan_ = plan;
  goal_pos_ = poseToPositionVector(current_plan_.back());
  goal_dir_ = poseToDirectionVector(current_plan_.back());
  // reset current and ahead face
  cancel_requested_ = false;
  current_face_ = lvr2::OptionalFaceHandle();
  return true;
}

bool RLMeshController::cancel()
{
  RCLCPP_INFO_STREAM(node_->get_logger(), "The RLMeshController has been requested to cancel!");
  cancel_requested_ = true;
  return true;
}

std::vector<geometry_msgs::msg::PoseStamped> RLMeshController::calculateLookahead(const std::vector<geometry_msgs::msg::PoseStamped>& plan, const std::vector<float>& distances)
{
  std::vector<geometry_msgs::msg::PoseStamped> lookahead_points;
  float accumulated_distance = 0.0;
  size_t distance_index = 0;

  for (size_t i = 1; i < plan.size() && distance_index < distances.size(); ++i)
  {
    const auto& prev_pose = plan[i - 1];
    const auto& curr_pose = plan[i];
    float segment_distance = std::sqrt(std::pow(curr_pose.pose.position.x - prev_pose.pose.position.x, 2) +
                                       std::pow(curr_pose.pose.position.y - prev_pose.pose.position.y, 2) +
                                       std::pow(curr_pose.pose.position.z - prev_pose.pose.position.z, 2));
    accumulated_distance += segment_distance;

    while (distance_index < distances.size() && accumulated_distance >= distances[distance_index])
    {
      lookahead_points.push_back(curr_pose);
      ++distance_index;
    }
  }

  return lookahead_points;
}

std::vector<float> RLMeshController::get_state()
{
  std::array<float, 12> state;
  state[0] = robot_pos_.x;
  state[1] = robot_pos_.y;
  state[2] = robot_pos_.z;
  state[3] = robot_dir_.x;
  state[4] = robot_dir_.y;
  state[5] = robot_dir_.z;

  std::vector<float> distances = {1.0, 2.0, 3.0};
  auto lookahead_points = calculateLookahead(current_plan_, distances);

  for (size_t i = 0; i < lookahead_points.size(); ++i)
  {
    state[6 + i * 3] = lookahead_points[i].pose.position.x;
    state[7 + i * 3] = lookahead_points[i].pose.position.y;
    state[8 + i * 3] = lookahead_points[i].pose.position.z;
  }

  RCLCPP_INFO(node_->get_logger(), "State: robot_pos: (%f, %f, %f), robot_dir: (%f, %f, %f), lookahead_points: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)",
              state[0], state[1], state[2], state[3], state[4], state[5],
              state[6], state[7], state[8], state[9], state[10], state[11]);

  return std::vector<float>(state.begin(), state.end());
}


mesh_map::Normal RLMeshController::poseToDirectionVector(const geometry_msgs::msg::PoseStamped& pose, const tf2::Vector3& axis)
{
  tf2::Transform transform;
  geometry_msgs::msg::Transform geom_transform;
  geom_transform.rotation = pose.pose.orientation;
  geom_transform.translation.x = pose.pose.position.x;
  geom_transform.translation.y = pose.pose.position.y;
  geom_transform.translation.z = pose.pose.position.z;
  tf2::fromMsg(geom_transform, transform);
  tf2::Vector3 v = transform.getBasis() * axis;
  return mesh_map::Normal(v.x(), v.y(), v.z());
}

mesh_map::Vector RLMeshController::poseToPositionVector(const geometry_msgs::msg::PoseStamped& pose)
{
  return mesh_map::Vector(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
}

float RLMeshController::gaussValue(const float& sigma_squared, const float& value)
{
  return exp(-value * value / 2 * sigma_squared) / sqrt(2 * M_PI * sigma_squared);
}

std::array<float, 2> RLMeshController::naiveControl(
    const mesh_map::Vector& robot_pos,
    const mesh_map::Normal& robot_dir,
    const mesh_map::Vector& mesh_dir,
    const mesh_map::Normal& mesh_normal,
    const float& mesh_cost)
{
  float phi = acos(mesh_dir.dot(robot_dir));
  float sign_phi = mesh_dir.cross(robot_dir).dot(mesh_normal);
  // debug output angle between supposed and current angle
  DEBUG_CALL(example_interfaces::msg::Float32 angle32; angle32.data = phi * 180 / M_PI; angle_pub_->publish(angle32);)

  float angular_velocity = copysignf(phi * config_.max_ang_velocity / M_PI, -sign_phi);
  const float max_angle = config_.max_angle * M_PI / 180.0;
  const float max_linear = config_.max_lin_velocity;
  float linear_velocity = phi <= max_angle ? max_linear - (phi * max_linear / max_angle) : 0.0;
  return { linear_velocity, angular_velocity };
}

rcl_interfaces::msg::SetParametersResult RLMeshController::reconfigureCallback(std::vector<rclcpp::Parameter> parameters)
{
  rcl_interfaces::msg::SetParametersResult result;

  for (auto parameter : parameters) {
    if (parameter.get_name() == name_ + ".max_lin_velocity") {
      config_.max_lin_velocity = parameter.as_double();
    } else if (parameter.get_name() == name_ + ".max_ang_velocity") {
      config_.max_ang_velocity= parameter.as_double();
    } else if (parameter.get_name() == name_ + ".arrival_fading") {
      config_.arrival_fading = parameter.as_double();
    } else if (parameter.get_name() == name_ + ".ang_vel_factor") {
      config_.ang_vel_factor = parameter.as_double();
    } else if (parameter.get_name() == name_ + ".lin_vel_factor") {
      config_.lin_vel_factor = parameter.as_double();
    } else if (parameter.get_name() == name_ + ".max_angle") {
      config_.max_angle = parameter.as_double();
    } else if (parameter.get_name() == name_ + ".max_search_radius") {
      config_.max_search_radius = parameter.as_double();
    } else if (parameter.get_name() == name_ + ".max_search_distance") {
      config_.max_search_distance = parameter.as_double();
    }
  }

  result.successful = true;
  return result;
}

bool RLMeshController::initialize(const std::string& plugin_name,
                                  const std::shared_ptr<tf2_ros::Buffer>& tf_ptr,
                                  const std::shared_ptr<mesh_map::MeshMap>& mesh_map_ptr,
                                  const rclcpp::Node::SharedPtr& node)
{
  node_ = node;
  map_ptr_ = mesh_map_ptr;
  name_ = plugin_name;

  angle_pub_ = node_->create_publisher<example_interfaces::msg::Float32>("~/current_angle", rclcpp::QoS(1).transient_local());

  { // cost max_lin_velocity
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = "Defines the maximum linear velocity";
    rcl_interfaces::msg::FloatingPointRange range;
    range.from_value = 0.0;
    range.to_value = 5.0;
    descriptor.floating_point_range.push_back(range);
    config_.max_lin_velocity = node->declare_parameter(name_ + ".max_lin_velocity", config_.max_lin_velocity);
  }
  { // cost max_ang_velocity
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = "Defines the maximum angular velocity";
    rcl_interfaces::msg::FloatingPointRange range;
    range.from_value = 0.0;
    range.to_value = 2.0;
    descriptor.floating_point_range.push_back(range);
    config_.max_ang_velocity = node->declare_parameter(name_ + ".max_ang_velocity", config_.max_ang_velocity);
  }
  { // cost arrival_fading
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = "Distance to goal position where the robot starts to fade down the linear velocity";
    rcl_interfaces::msg::FloatingPointRange range;
    range.from_value = 0.0;
    range.to_value = 5.0;
    descriptor.floating_point_range.push_back(range);
    config_.arrival_fading = node->declare_parameter(name_ + ".arrival_fading", config_.arrival_fading);
  }
  { // cost ang_vel_factor
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = "Factor for angular velocity";
    rcl_interfaces::msg::FloatingPointRange range;
    range.from_value = 0.1;
    range.to_value = 10.0;
    descriptor.floating_point_range.push_back(range);
    config_.ang_vel_factor = node->declare_parameter(name_ + ".ang_vel_factor", config_.ang_vel_factor);
  }
  { // cost lin_vel_factor
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = "Factor for linear velocity";
    rcl_interfaces::msg::FloatingPointRange range;
    range.from_value = 0.1;
    range.to_value = 10.0;
    descriptor.floating_point_range.push_back(range);
    config_.lin_vel_factor = node->declare_parameter(name_ + ".lin_vel_factor", config_.lin_vel_factor);
  }
  { // cost max_angle
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = "The maximum angle for the linear velocity function";
    rcl_interfaces::msg::FloatingPointRange range;
    range.from_value = 1.0;
    range.to_value = 180.0;
    descriptor.floating_point_range.push_back(range);
    config_.max_angle = node->declare_parameter(name_ + ".max_angle", config_.max_angle);
  }
  { // cost max_search_radius
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = "The maximum radius in which to search for a consecutive neighbour face";
    rcl_interfaces::msg::FloatingPointRange range;
    range.from_value = 0.01;
    range.to_value = 2.0;
    descriptor.floating_point_range.push_back(range);
    config_.max_search_radius = node->declare_parameter(name_ + ".max_search_radius", config_.max_search_radius);
  }
  { // cost max_search_distance
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = "The maximum distance from the surface which is accepted for projection";
    rcl_interfaces::msg::FloatingPointRange range;
    range.from_value = 0.01;
    range.to_value = 2.0;
    descriptor.floating_point_range.push_back(range);
    config_.max_search_distance = node->declare_parameter(name_ + ".max_search_distance", config_.max_search_distance);
  }

  reconfiguration_callback_handle_ = node_->add_on_set_parameters_callback(std::bind(
      &RLMeshController::reconfigureCallback, this, std::placeholders::_1));

  RCLCPP_INFO(node_->get_logger(), "Init the rest of the controller...");
        // Create a new TensorFlow session
        // Create a new TensorFlow session
        tensorflow::SessionOptions options;
        tensorflow::Status status = tensorflow::NewSession(options, &session_);
        if (!status.ok()) {
            throw std::runtime_error("Failed to create TensorFlow session: " + status.ToString());
        }

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

        // Define the critic network
        tensorflow::NodeDef* critic_input = graph_def.add_node();
        critic_input->set_name("critic_input");
        critic_input->set_op("Placeholder");
        (*critic_input->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*critic_input->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(-1);
        (*critic_input->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(15);

        tensorflow::NodeDef* critic_fc1_weights = graph_def.add_node();
        critic_fc1_weights->set_name("critic_fc1_weights");
        critic_fc1_weights->set_op("Const");
        (*critic_fc1_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*critic_fc1_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
        (*critic_fc1_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(15);
        (*critic_fc1_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);

        tensorflow::NodeDef* critic_fc1_bias_weights = graph_def.add_node();
        critic_fc1_bias_weights->set_name("critic_fc1_bias_weights");
        critic_fc1_bias_weights->set_op("Const");
        (*critic_fc1_bias_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*critic_fc1_bias_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
        (*critic_fc1_bias_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);

        tensorflow::NodeDef* critic_fc1 = graph_def.add_node();
        critic_fc1->set_name("critic_fc1");
        critic_fc1->set_op("MatMul");
        (*critic_fc1->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        critic_fc1->add_input("critic_input");
        critic_fc1->add_input("critic_fc1_weights");

        tensorflow::NodeDef* critic_fc1_bias = graph_def.add_node();
        critic_fc1_bias->set_name("critic_fc1_bias");
        critic_fc1_bias->set_op("BiasAdd");
        (*critic_fc1_bias->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        critic_fc1_bias->add_input("critic_fc1");
        critic_fc1_bias->add_input("critic_fc1_bias_weights");

        tensorflow::NodeDef* critic_fc1_relu = graph_def.add_node();
        critic_fc1_relu->set_name("critic_fc1_relu");
        critic_fc1_relu->set_op("Relu");
        (*critic_fc1_relu->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        critic_fc1_relu->add_input("critic_fc1_bias");

        tensorflow::NodeDef* critic_fc2_weights = graph_def.add_node();
        critic_fc2_weights->set_name("critic_fc2_weights");
        critic_fc2_weights->set_op("Const");
        (*critic_fc2_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*critic_fc2_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
        (*critic_fc2_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);
        (*critic_fc2_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);

        tensorflow::NodeDef* critic_fc2_bias_weights = graph_def.add_node();
        critic_fc2_bias_weights->set_name("critic_fc2_bias_weights");
        critic_fc2_bias_weights->set_op("Const");
        (*critic_fc2_bias_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*critic_fc2_bias_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
        (*critic_fc2_bias_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);

        tensorflow::NodeDef* critic_fc2 = graph_def.add_node();
        critic_fc2->set_name("critic_fc2");
        critic_fc2->set_op("MatMul");
        (*critic_fc2->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        critic_fc2->add_input("critic_fc1_relu");
        critic_fc2->add_input("critic_fc2_weights");

        tensorflow::NodeDef* critic_fc2_bias = graph_def.add_node();
        critic_fc2_bias->set_name("critic_fc2_bias");
        critic_fc2_bias->set_op("BiasAdd");
        (*critic_fc2_bias->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        critic_fc2_bias->add_input("critic_fc2");
        critic_fc2_bias->add_input("critic_fc2_bias_weights");

        tensorflow::NodeDef* critic_fc2_relu = graph_def.add_node();
        critic_fc2_relu->set_name("critic_fc2_relu");
        critic_fc2_relu->set_op("Relu");
        (*critic_fc2_relu->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        critic_fc2_relu->add_input("critic_fc2_bias");

        tensorflow::NodeDef* critic_output_weights = graph_def.add_node();
        critic_output_weights->set_name("critic_output_weights");
        critic_output_weights->set_op("Const");
        (*critic_output_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*critic_output_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
        (*critic_output_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(256);
        (*critic_output_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(1);

        tensorflow::NodeDef* critic_output_bias_weights = graph_def.add_node();
        critic_output_bias_weights->set_name("critic_output_bias_weights");
        critic_output_bias_weights->set_op("Const");
        (*critic_output_bias_weights->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*critic_output_bias_weights->mutable_attr())["value"].mutable_tensor()->set_dtype(tensorflow::DT_FLOAT);
        (*critic_output_bias_weights->mutable_attr())["value"].mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(1);

        tensorflow::NodeDef* critic_output = graph_def.add_node();
        critic_output->set_name("critic_output");
        critic_output->set_op("MatMul");
        (*critic_output->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        critic_output->add_input("critic_fc2_relu");
        critic_output->add_input("critic_output_weights");

        tensorflow::NodeDef* critic_output_bias = graph_def.add_node();
        critic_output_bias->set_name("critic_output_bias");
        critic_output_bias->set_op("BiasAdd");
        (*critic_output_bias->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        critic_output_bias->add_input("critic_output");
        critic_output_bias->add_input("critic_output_bias_weights");

        // Create the graph in the session
        status = session_->Create(graph_def);
        if (!status.ok()) {
            throw std::runtime_error("Failed to create graph: " + status.ToString());
        }

  return true;
}
} /* namespace mesh_controller */