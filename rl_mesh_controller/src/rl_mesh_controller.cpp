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
#include <cmath>

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

  // Ensure the state vector has exactly 12 elements
  if (state.size() != 10) {
    RCLCPP_ERROR(node_->get_logger(), "State vector size is not 9, it is %zu", state.size());
    return mbf_msgs::action::ExePath::Result::FAILURE;
  }
  // Publish the state
  auto state_msg = std_msgs::msg::Float32MultiArray();
  state_msg.data = state;
  float reward = 0.0f;
  state_publisher_->publish(state_msg);
   // Or a neutral value
  // 2. Calculate reward and observe next state based on the previous state and action
  if (!previous_state_.empty() && !previous_action_.empty() && !std::isnan(previous_reward_)) {
      float initial_distance = last_goal_distance_;


      // Calculate new distance
      float new_distance = (goal_pos_ - robot_pos_).length();

      //float de_penalty = (initial_distance - new_distance)*state[0];
      float distance_diff = initial_distance - new_distance;
      float scale_factor = 1000.0f;  // Adjust this value to amplify the effect
      float scaled_distance_diff = distance_diff * scale_factor;
      float de_penalty = std::exp(scaled_distance_diff) * distance_diff * (distance_diff > 0 ? 1 : -1) * state[0];
      // Heading Error (HE)
      float he = previous_state_[2];

      // Heading Error (HE) Penalty
      //float he_penalty = he / (1 + new_distance);  // Scale down HE penalty when distance error increases
      float Kh = .5f;  // Tune this value
      float Sh = .5f;  // Shape factor for scaling
      float distance_weight = std::min(1.0f, 1.0f / (new_distance + 0.1f));  // Avoid division by zero
      // float he_penalty = -Kh * pow(std::abs(he), Sh) * distance_weight;
      float he_penalty = -Kh * pow(std::abs(he) / (1 + new_distance), Sh);
      // Total reward
      reward = de_penalty + he_penalty;

      float min_reward = -0.1f;
      float max_reward = 0.1f;

      // Ensure reward stays within bounds before normalizing
      //reward = std::max(min_reward, std::min(reward, max_reward));

      // Apply normalization
      //reward = ((reward - min_reward) / (max_reward - min_reward)) * 200.0f - 100.0f;
      // if (new_distance < initial_distance) {
      //   reward += 0.5f * (initial_distance - new_distance);  // Boost positive rewards
    std::vector<float> next_state = state;  // Update based on action
    auto msg = rl_mesh_controller_msgs::msg::StateActionRewardNextState();
    msg.state = previous_state_;
    msg.action = previous_action_;
    msg.reward = reward;
    msg.next_state = state;
    state_msg_ = msg;
    // Define exclusion radius
    float exclusion_radius = 0.2f;  // Adjust as needed
    float reward_change_threshold = 2.0f;
    float reward_diff = std::abs(reward - previous_reward_);
    if (!std::isnan(previous_reward_) && reward_diff > reward_change_threshold) {
        RCLCPP_WARN(node_->get_logger(), "Skipping state: sudden reward change detected");
    } else {
      state_buffer_publisher_->publish(msg);
    }
  }
  //RCLCPP_INFO(node_->get_logger(), "Previous Reward: %f, Current Reward: %f", previous_reward_, reward);
  // 3. Select action (inference or exploration)
  std::vector<float> action(3);

  // 4. Execute action
  if (received_twist_) {
      cmd_vel.twist.linear.x = std::min(config_.max_lin_velocity, received_twist_->linear.x * config_.lin_vel_factor);
      cmd_vel.twist.linear.y = std::min(config_.max_lin_velocity, received_twist_->linear.y * config_.lin_vel_factor);
      cmd_vel.twist.angular.z = std::min(config_.max_ang_velocity, received_twist_->angular.z * config_.ang_vel_factor);
      cmd_vel.header.stamp = node_->now();
      action[0] = cmd_vel.twist.linear.x;
      action[1] = cmd_vel.twist.linear.y;
      action[2] = cmd_vel.twist.angular.z;
  }

  // 5. Update previous state and action
    previous_state_ = state;
    previous_action_ = action;
    previous_reward_ = reward;
    last_goal_distance_ = (goal_pos_ - robot_pos_).length();


  if (cancel_requested_)
  {
    return mbf_msgs::action::ExePath::Result::CANCELED;
  }
  return mbf_msgs::action::ExePath::Result::SUCCESS;
}

void RLMeshController::tensorActionCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    received_twist_ = msg;
}
void RLMeshController::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
   //RCLCPP_INFO(node_->get_logger(), "Received odometry: angular velocity z: %f", msg->twist.twist.angular.z);
    angular_velocity_ = msg->twist.twist.angular.z;

}

void RLMeshController::trainModel()
{
  RCLCPP_INFO(node_->get_logger(), "Starting training model");
  exploration_threshold_ *= 0.8;  // Decay exploration threshold
}

bool RLMeshController::isGoalReached(double dist_tolerance, double angle_tolerance)
{
  float goal_distance = (goal_pos_ - robot_pos_).length();
  float angle = acos(goal_dir_.dot(robot_dir_));
  goal_reached_ = goal_distance <= static_cast<float>(dist_tolerance) && angle <= static_cast<float>(angle_tolerance);
  if (goal_reached_){
    state_msg_.reward += 5;  // Reward for reaching the goal
    if (state_msg_.reward >10)
      state_msg_.reward = 10;
    state_buffer_publisher_->publish(state_msg_);
  }
  return goal_reached_;
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
  std::array<float, 10> state;
  state.fill(1.0f);  // Initialize the array with 1.0 floats

  // Calculate Distance Error (DE)
  float de = std::sqrt(std::pow(goal_pos_.x - robot_pos_.x, 2) +
                        std::pow(goal_pos_.y - robot_pos_.y, 2) +
                        std::pow(goal_pos_.z - robot_pos_.z, 2));

  // Calculate Derivative of Distance Error (DDE)
  rclcpp::Time current_time = node_->now();
  float time_step = (current_time - previous_time_).seconds();
  float dde;
  if (time_step <= 0.0f) {
    time_step = 1.0;
  }
  dde = (de - previous_de_) / time_step;
  // Update previous DE and previous time
  previous_de_ = de;
  previous_time_ = current_time;
  float dot_product = goal_dir_.x * robot_dir_.x +
                      goal_dir_.y * robot_dir_.y +
                      goal_dir_.z * robot_dir_.z;

  float goal_dir_mag = std::sqrt(goal_dir_.x * goal_dir_.x +
                                goal_dir_.y * goal_dir_.y +
                                goal_dir_.z * goal_dir_.z);

  float robot_dir_mag = std::sqrt(robot_dir_.x * robot_dir_.x +
                                  robot_dir_.y * robot_dir_.y +
                                  robot_dir_.z * robot_dir_.z);

  float cos_theta = dot_product / (goal_dir_mag * robot_dir_mag);
  float he_magnitude = std::acos(std::clamp(cos_theta, -1.0f, 1.0f)); // Angle in radians
  // Calculate Heading Error (HE)
  geometry_msgs::msg::Vector3 he;
  he.x = goal_dir_.x - robot_dir_.x;
  he.y = goal_dir_.y - robot_dir_.y;
  he.z = goal_dir_.z - robot_dir_.z;

  // Set DE, DDE, and HE as the first elements of the state
  state[0] = std::round(de);
  state[1] = dde;
  state[2] = he_magnitude;//std::sqrt(std::pow(he.x, 2) + std::pow(he.y, 2) + std::pow(he.z, 2)); //he_magnitude// Magnitude of HE

  std::vector<float> distances = {1.0, 2.0, 3.0};
  auto lookahead_points = calculateLookahead(current_plan_, distances);

  if (lookahead_points.size() != 3) {
    RCLCPP_ERROR(node_->get_logger(), "Lookahead points size is not 3, it is %zu", lookahead_points.size());
    return std::vector<float>(state.begin(), state.end());
  }

  for (size_t i = 0; i < lookahead_points.size(); ++i)
  {
    // Calculate Distance to Look-Ahead Point (Dl)
    float dx = lookahead_points[i].pose.position.x - robot_pos_.x;
    float dy = lookahead_points[i].pose.position.y - robot_pos_.y;
    float dz = lookahead_points[i].pose.position.z - robot_pos_.z;
    float dl = std::sqrt(dx * dx + dy * dy + dz * dz);

    // Calculate Angle to Look-Ahead Point (θl)
    float dot_product = dx * robot_dir_.x + dy * robot_dir_.y + dz * robot_dir_.z;
    float robot_dir_magnitude = std::sqrt(robot_dir_.x * robot_dir_.x + robot_dir_.y * robot_dir_.y + robot_dir_.z * robot_dir_.z);
    float lookahead_dir_magnitude = std::sqrt(dx * dx + dy * dy + dz * dz);
    float cos_theta = dot_product / (robot_dir_magnitude * lookahead_dir_magnitude);
    float theta = std::acos(cos_theta);

    state[3 + i * 2] = std::round(dl);
    state[4 + i * 2] = theta;
  }

  state[9] = angular_velocity_;  // Angular velocity
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

  state_publisher_ = node_->create_publisher<std_msgs::msg::Float32MultiArray>("/model_state", 10);
  tensor_action_subscription_ = node_->create_subscription<geometry_msgs::msg::Twist>(
    "/tensor_action", 10, std::bind(&RLMeshController::tensorActionCallback, this, std::placeholders::_1));
  odom_subscription_ = node_->create_subscription<nav_msgs::msg::Odometry>(
    "/Spot/odometry", 10, std::bind(&RLMeshController::odomCallback, this, std::placeholders::_1));
  state_buffer_publisher_ = node_->create_publisher<rl_mesh_controller_msgs::msg::StateActionRewardNextState>("/state_buffer", 10);
  exploration_threshold_ = 40.0;  // 40% exploration
  float previous_reward_ = std::numeric_limits<float>::quiet_NaN();
  previous_de_ = 0.0;
  previous_time_ = node_->now();
  angular_velocity_= 0.0f;


  return true;
}
} /* namespace mesh_controller */