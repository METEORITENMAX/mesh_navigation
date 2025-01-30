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
    if (state.size() != 12) {
    RCLCPP_ERROR(node_->get_logger(), "State vector size is not 12, it is %zu", state.size());
    return mbf_msgs::action::ExePath::Result::FAILURE;
  }
  // 2. Calculate reward and observe next state based on the previous state and action
  if (!previous_state_.empty() && !previous_action_.empty()) {
    float goal_distance = (goal_pos_ - robot_pos_).length();
    float reward = goal_distance;  // Define reward function as goal distance.
    std::vector<float> next_state = state;  // Update based on action.

    // Store transition in replay buffer
    if (replay_buffer_.size() >= replay_buffer_size_) {
      replay_buffer_.erase(replay_buffer_.begin());
    }
    replay_buffer_.emplace_back(previous_state_, previous_action_, reward, next_state);
  }

  // 3. Select action (inference or exploration)
  std::vector<float> action(3);
  if (training_mode_ && rand() % 100 < 20) {  // 20% exploration
    action[0] = ((float)rand() / RAND_MAX) * config_.max_lin_velocity;
    action[1] = ((float)rand() / RAND_MAX) * config_.max_lin_velocity;
    action[2] = ((float)rand() / RAND_MAX) * config_.max_ang_velocity;
  } else {
    tensorflow::Tensor state_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, static_cast<long>(state.size())}));
    std::copy(state.begin(), state.end(), state_tensor.flat<float>().data());
    std::vector<tensorflow::Tensor> outputs = actor_critic_network_->Predict(state_tensor);
    auto actor_output = outputs[0].flat<float>();
    action[0] = actor_output(0) * config_.max_lin_velocity;
    action[1] = actor_output(1) * config_.max_lin_velocity;
    action[2] = actor_output(2) * config_.max_ang_velocity;
  }

  // 4. Execute action
  cmd_vel.twist.linear.x = std::min(config_.max_lin_velocity, action[0] * config_.lin_vel_factor);
  cmd_vel.twist.linear.y = std::min(config_.max_lin_velocity, action[1] * config_.lin_vel_factor);
  cmd_vel.twist.angular.z = std::min(config_.max_ang_velocity, action[2] * config_.ang_vel_factor);
  cmd_vel.header.stamp = node_->now();

  // 5. Update previous state and action
  previous_state_ = state;
  previous_action_ = action;


  // 6. Train model if in training mode
  if (training_mode_ && replay_buffer_.size() > batch_size_) {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Training Model");
    trainModel();
  }

  if (cancel_requested_)
  {
    return mbf_msgs::action::ExePath::Result::CANCELED;
  }
  return mbf_msgs::action::ExePath::Result::SUCCESS;
}

void RLMeshController::trainModel()
{
  // Sample a batch of transitions from the replay buffer
  std::vector<std::tuple<std::vector<float>, std::vector<float>, float, std::vector<float>>> batch;
  std::sample(replay_buffer_.begin(), replay_buffer_.end(), std::back_inserter(batch), batch_size_, std::mt19937{std::random_device{}()});

  // Prepare tensors for states, actions, rewards, and next states
  tensorflow::Tensor states(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<long>(batch.size()), 12}));
  tensorflow::Tensor actions(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<long>(batch.size()), 3}));
  tensorflow::Tensor rewards(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<long>(batch.size()), 1}));
  tensorflow::Tensor next_states(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<long>(batch.size()), 12}));

  for (size_t i = 0; i < batch.size(); ++i) {
    const auto& [state, action, reward, next_state] = batch[i];
    std::copy(state.begin(), state.end(), &states.matrix<float>()(i, 0));
    std::copy(action.begin(), action.end(), &actions.matrix<float>()(i, 0));
    rewards.matrix<float>()(i, 0) = reward;
    std::copy(next_state.begin(), next_state.end(), &next_states.matrix<float>()(i, 0));
  }

  // Train the actor-critic network
  actor_critic_network_->Train(states, actions, rewards, next_states);

  // Wipe replay buffer
  replay_buffer_.clear();
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
    // RCLCPP_INFO(node_->get_logger(), "Pose %zu: Position (x: %f, y: %f, z: %f), Orientation (x: %f, y: %f, z: %f, w: %f)",
    //             i,
    //             pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
    //             pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w);
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

  // Ensure that we always return exactly 3 points
  while (lookahead_points.size() < 3)
  {
    if (!lookahead_points.empty())
    {
      lookahead_points.push_back(lookahead_points.back());
    }
    else if (!plan.empty())
    {
      lookahead_points.push_back(plan.back());
    }
    else
    {
      // If the plan is empty, return a default PoseStamped
      geometry_msgs::msg::PoseStamped default_pose;
      lookahead_points.push_back(default_pose);
    }
  }

  // If there are more than 3 points, truncate the list to 3 points
  if (lookahead_points.size() > 3)
  {
    lookahead_points.resize(3);
  }

  return lookahead_points;
}

std::vector<float> RLMeshController::get_state()
{
  std::array<float, 12> state;
  state.fill(1.0f);  // Initialize the array with 1.0 floats
  state[0] = robot_pos_.x;
  state[1] = robot_pos_.y;
  state[2] = robot_pos_.z;
  state[3] = robot_dir_.x;
  state[4] = robot_dir_.y;
  state[5] = robot_dir_.z;

  std::vector<float> distances = {1.0, 2.0, 3.0};
  auto lookahead_points = calculateLookahead(current_plan_, distances);

  if (lookahead_points.size() != 3) {
    RCLCPP_ERROR(node_->get_logger(), "Lookahead points size is not 3, it is %zu", lookahead_points.size());
    return std::vector<float>(state.begin(), state.end());
  }

  for (size_t i = 0; i < lookahead_points.size(); ++i)
  {
    state[6 + i * 3] = lookahead_points[i].pose.position.x;
    state[7 + i * 3] = lookahead_points[i].pose.position.y;
    state[8 + i * 3] = lookahead_points[i].pose.position.z;
  }
  //print state size
  //RCLCPP_INFO(node_->get_logger(), "state size: %d", state.size());
  // RCLCPP_INFO(node_->get_logger(), "State: robot_pos: (%f, %f, %f), robot_dir: (%f, %f, %f), lookahead_points: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)",
  //             state[0], state[1], state[2], state[3], state[4], state[5],
  //             state[6], state[7], state[8], state[9], state[10], state[11]);

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

  RCLCPP_INFO(node_->get_logger(), "Initialized Actor Critic Network");
  // Initialize the actor-critic network
  actor_critic_network_ = std::make_unique<ActorCriticNetwork>();
  actor_critic_network_->initializeGraph();


  return true;
}
} /* namespace mesh_controller */