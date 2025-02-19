import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from rl_mesh_controller_msgs.msg import StateActionRewardNextState
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from .replay_buffer import ReplayBuffer
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
""" action_dim: dimensions for which the noise will be generated
    mu: mean value towards which the noise will tend to revert over time.
    theta: how quickly the noise reverts to the mean (mu)
    sigma: scale of the noise
"""
class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck noise process"""
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.01, dt=1e-2):
        self.action_dim = action_dim  # Dimensionality of action space
        self.mu = mu  # Mean
        self.theta = theta  # Mean reversion
        self.sigma = sigma  # Volatility
        self.dt = dt  # Time step
        self.state = torch.zeros(action_dim)  # Initial noise state (zeros)

    def reset(self):
        """Reset the noise process state"""
        self.state = torch.zeros(self.action_dim)

    def __call__(self):
        """Generate noise based on the OU process"""
        dt_tensor = torch.tensor(self.dt, dtype=torch.float32)  # Convert dt to a tensor
        noise = self.theta * (self.mu - self.state) * dt_tensor + self.sigma * torch.sqrt(dt_tensor) * torch.randn_like(self.state)
        self.state = self.state + noise
        return self.state

class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)  # Third fully connected layer (output layer)

        self.log_std = nn.Parameter(torch.zeros(output_size))  # Learnable log standard deviation
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Xavier initialization for the first fully connected layer
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        # Xavier initialization for the second fully connected layer
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_mean = torch.tanh(self.fc3(x))   # Mean of action distribution
        action_log_std = self.log_std.expand_as(action_mean)  # Log standard deviation
        return action_mean, action_log_std

class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = nn.LeakyReLU(0.1)(self.fc1(x))
        x = self.fc2(x)
        return x

class ActorCriticNode(Node):
    def __init__(self):
        super().__init__('actor_critic_node')
        self.state_subscription = self.create_subscription(
            Float32MultiArray,
            '/model_state',
            self.state_callback,
            10)
        self.state_subscription  # prevent unused variable warning


        self.state_buffer_subscription = self.create_subscription(
            StateActionRewardNextState,
            '/state_buffer',
            self.state_buffer_callback,
            10)
        self.state_buffer_subscription  # prevent unused variable warning

        self.tensor_action_publisher = self.create_publisher(Twist, '/tensor_action', 10)

        # Initialize the actor and critic networks
        input_size = 10
        hidden_size = 64
        output_size = 3
        self.actor = ActorNetwork(input_size, hidden_size, output_size)
        self.noise_process = OrnsteinUhlenbeckNoise(action_dim=3, mu=0, theta=0.15, sigma=0.4)
        self.target_actor = ActorNetwork(input_size, hidden_size, output_size)
        self.critic = CriticNetwork(input_size+output_size, hidden_size)
        self.target_critic = CriticNetwork(input_size+output_size, hidden_size)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.eval()
        self.target_critic.eval()
        self.get_logger().info('ActorCriticNode initialized.')

        # Define the loss function and optimizers
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        # Discount factor for future rewards
        self.gamma = 0.9

        # Initialize exploration chance
        self.exploration_chance = 0.0
        self.replay_buffer = ReplayBuffer(capacity=10000)
        #self.replay_buffer_neg = ReplayBuffer(capacity=4096)
        self.long_term_buffer_neg = ReplayBuffer(capacity=10000)
        self.long_term_buffer_pos = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.longterm_batch_size = 500
        #Create timers
        self.create_timer(2., self.train)
        self.create_timer(30.0, self.reset_uhlennoise)
        # Exploration phase settings
        self.max_exploration_time = 300  # Stop after 5 minutes (300 sec)
        self.exploration_start_time = time.time()
        self.saved_motions_active = True
        self.exploration_active = True
    def reset_uhlennoise(self):
        self.noise_process.reset()
    def state_callback(self, msg):
        #if not self.saved_motions_active:
        # Convert the received data to a tensor
        input_tensor = torch.tensor(msg.data, dtype=torch.float32).view(1, -1)
        #self.get_logger().info(f'Received input tensor: {input_tensor}')

        normalized_input = self.normalize_input(input_tensor)
        #self.get_logger().info(f'Normalized input tensor: {normalized_input}')

        # Forward pass through the actor network
        output_tensor = self.actor(normalized_input)[0]
        noise=self.noise_process()
        output_tensor = output_tensor[0]+ noise

        twist_msg = Twist()
        twist_msg.linear.x = output_tensor[0].item()
        twist_msg.linear.y = output_tensor[1].item()
        twist_msg.angular.z = output_tensor[2].item()
        self.tensor_action_publisher.publish(twist_msg)
        self.get_logger().info(f'noise: {noise}')
    """
    0: de
    1: dee
    2: he
    3,5,7: distance to lookahead
    4,6,8: magnitude to lookahead
    9: angular velocity
    """
    def normalize_input(self, input_tensor):
        #Define min and max values for each feature
        # input_np = input_tensor.numpy()
        # # Min-Max Scaler for distances and angles
        # scaler_minmax = MinMaxScaler(feature_range=(-10, 10))
        # input_np[:, [0, 3, 5, 7]] = scaler_minmax.fit_transform(input_np[:, [0, 3, 5, 7]])  # Distances

        # # Standard Scaler for derivatives and angular velocity
        # scaler_std = StandardScaler()
        # input_np[:, [1, 2, 4, 6, 8, 9]] = scaler_std.fit_transform(input_np[:, [1, 2, 4, 6 ,8, 9]])

        return input_tensor#torch.tensor(input_np, dtype=torch.float32)

    def state_buffer_callback(self, msg):
        # Process the StateActionRewardNextState message


        state = torch.tensor(msg.state, dtype=torch.float32).view(1, -1)
        action = torch.tensor(msg.action, dtype=torch.float32).view(1, -1)
        reward = torch.tensor([msg.reward], dtype=torch.float32)
        min_reward_threshold = -10  # Set your threshold
        next_state = torch.tensor(msg.next_state, dtype=torch.float32).view(1, -1)

        # if 0.0 > msg.reward > min_reward_threshold or abs(msg.reward) > 100.0:
        #     return
        # Normalize the state and next_state tensors
        normalized_state = self.normalize_input(state)
        normalized_next_state = self.normalize_input(next_state)

        #reward = (reward - min_reward) / (max_reward - min_reward)


        reward *= 100

        self.get_logger().info(f'reward: {normalized_state}')
        if abs(reward) > .6:
            if reward > .0:
                self.long_term_buffer_pos.push(normalized_state, action, reward, normalized_next_state)
            else:
                self.long_term_buffer_neg.push(normalized_state, action, reward, normalized_next_state)

        self.replay_buffer.push(normalized_state, action, reward, normalized_next_state)

        # Train if the buffer is large enough
        # if len(self.replay_buffer) >= self.batch_size:
        #     self.train()

    def train(self):
        self.get_logger().info(f'+++++++++++++++++++++++++++++++++++++++******TRAINING******+++++++++++++++++++++++++++++++++++++++')

        # Ensure the replay buffer has enough data before training
        if len(self.replay_buffer) < self.batch_size:
            self.get_logger().info(f'Not enough data in replay buffer. Size: {len(self.replay_buffer)}')
            return

        # Sample from the live replay buffer first
        batch_size_live = min(self.batch_size, len(self.replay_buffer))
        state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample(batch_size_live)

        batch_size_pos = min(self.longterm_batch_size // 2, len(self.long_term_buffer_pos))
        batch_size_neg = min(self.longterm_batch_size // 2, len(self.long_term_buffer_neg))

        if batch_size_pos > 0:
            state_batch_pos, action_batch_pos, reward_batch_pos, next_state_batch_pos = self.long_term_buffer_pos.sample(batch_size_pos)
        if batch_size_neg > 0:
            state_batch_neg, action_batch_neg, reward_batch_neg, next_state_batch_neg = self.long_term_buffer_neg.sample(batch_size_neg)

        # Combine all batches together
        if batch_size_pos > 0 and batch_size_neg > 0:
            state_batch = torch.cat([state_batch, state_batch_pos, state_batch_neg], dim=0)
            action_batch = torch.cat([action_batch, action_batch_pos, action_batch_neg], dim=0)
            reward_batch = torch.cat([reward_batch, reward_batch_pos, reward_batch_neg], dim=0).view(-1, 1)
            next_state_batch = torch.cat([next_state_batch, next_state_batch_pos, next_state_batch_neg], dim=0)
            self.get_logger().info(f'Using both buffers: {self.batch_size} from live + {self.longterm_batch_size} from long-term')

        reward_batch = reward_batch.view(-1, 1)

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.)
        next_action_batch = self.target_actor(next_state_batch)
        next_state_action_batch = torch.cat([next_state_batch, next_action_batch[0]], dim=1)
        #target_value_batch = reward_batch + self.gamma * self.target_critic(next_state_action_batch)
        target_value_batch = reward_batch + self.gamma * self.target_critic(next_state_action_batch).detach()

        # Compute critic loss
        state_action_batch = torch.cat([state_batch, action_batch], dim=1)
        predicted_value_batch = self.critic(state_action_batch)
        critic_loss = self.critic_criterion(predicted_value_batch, target_value_batch)
        actor_loss = -self.critic(torch.cat([state_batch, self.actor(state_batch)[0]], dim=1)).mean()

        action_mean, action_log_std = self.actor(state_batch)
        action_log_std = torch.clamp(action_log_std, min=-.5, max=.5)
        action_std = action_log_std.exp()
        action_std = torch.clamp(action_std, min=1e-6)
        #self.get_logger().info(f'Action std: {action_std}')

        probs = torch.distributions.Normal(action_mean, action_std).sample()
        probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6)
        expected_return = self.critic(torch.cat([state_batch, self.actor(state_batch)[0]], dim=1)).mean()
        alpha = 1e-3  # Adjust based on your entropy scaling needs
        epsilon = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1).mean()
        #self.get_logger().info(f"Action probs: {probs}")
        #self.get_logger().info(f"Action entropy: {entropy}")
        #actor_loss = -expected_return + alpha * entropy
        self.get_logger().info(f'-------------------------Computed critic loss: {critic_loss.item()}')
        self.get_logger().info(f'-------------------------Computed actor loss: {actor_loss.item()}')
        critic_loss.backward()
        actor_loss.backward()
        #total_loss = critic_loss + actor_loss

        # Zero out gradients before backpropagation
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Backpropagate only once
        #total_loss.backward()

        # Update both networks
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        # self.get_logger().info(f'Target value batch: {target_value_batch}')
        # self.get_logger().info(f'Predicted critic values: {predicted_value_batch}')
        # Update target networks
        self.update_target_networks()

    def update_target_networks(self):
        tau = 0.005  # Small update factor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, file_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_input_size': 10,  # State size for actor
            'critic_input_size': 13,  # State size + action size for critic
            'output_size': 3
        }, file_path)
        self.get_logger().info(f'Model saved to {file_path}')

    def load_model(self, file_path):
        try:
            checkpoint = torch.load(file_path, weights_only=True)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.get_logger().info(f'Model loaded from {file_path}')
        except KeyError as e:
            self.get_logger().error(f'Missing key in checkpoint: {e}')
        except FileNotFoundError:
            self.get_logger().warning(f'Checkpoint file not found: {file_path}')

def main(args=None):
    rclpy.init(args=args)
    node = ActorCriticNode()
    try:
        node.load_model('/home/max/tensor_models/pytorch/model.pth')  # Load the model if it exists
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_model('/home/max/tensor_models/pytorch/model.pth')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()