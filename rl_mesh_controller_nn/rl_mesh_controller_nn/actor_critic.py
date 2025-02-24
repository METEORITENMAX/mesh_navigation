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

import threading
import matplotlib.pyplot as plt



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

class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ValueNetwork, self).__init__()
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

class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)

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

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_mean = torch.tanh(self.fc2(x))   # Mean of action distribution
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
        self.sigma = 0.1
        self.theta = 0.15
        self.noise_process = OrnsteinUhlenbeckNoise(action_dim=3, mu=0, theta=self.theta, sigma=self.sigma)

        self.actor = ActorNetwork(input_size, hidden_size, output_size)
        self.target_actor = ActorNetwork(input_size, hidden_size, output_size)
        self.critic = CriticNetwork(input_size+output_size, hidden_size)
        self.target_critic = CriticNetwork(input_size+output_size, hidden_size)
        self.value = ValueNetwork(input_size, hidden_size)
        self.target_value = ValueNetwork(input_size, hidden_size)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_value.load_state_dict(self.value.state_dict())

        self.target_actor.eval()
        self.target_critic.eval()
        self.target_value.eval()
        self.get_logger().info('ActorCriticNode initialized.')

        # Define the loss function and optimizers
        self.lr = 1e-7
        self.critic_criterion = nn.MSELoss()
        self.value_criterion = nn.MSELoss()
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)


        # Discount factor for future rewards
        self.gamma = 0.9

        # Initialize exploration chance
        self.exploration_chance = 0.0
        self.replay_buffer = ReplayBuffer(capacity=10000)
        #self.replay_buffer_neg = ReplayBuffer(capacity=4096)
        self.long_term_buffer_neg = ReplayBuffer(capacity=10000)
        self.long_term_buffer_pos = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.longterm_batch_size = 500
        #Create timers
        self.create_timer(2., self.train)
        self.create_timer(30.0, self.reset_uhlennoise)
        # Exploration phase settings
        self.max_exploration_time = 300  # Stop after 5 minutes (300 sec)
        self.exploration_start_time = time.time()
        # Initialize lists to store loss values
        self.actor_losses = []
        self.critic_losses = []
        self.time_steps = []
        self.output_tensor = []
        self.output_tensor_noise = []
        self.de_list = []
        self.he_list = []
        self.reward_list = []
        self.tau = 0.005
        self.global_params = {
            'Learning Rate': self.lr ,
            'Batch Size': self.batch_size,
            'Gamma': self.gamma,
            'Tau': self.tau,
            'Sigma': self.sigma,
            'Theta': self.theta
        }
        # Start the plotting thread
        self.plotting_thread = threading.Thread(target=self.plot_losses)
        self.plotting_thread.start()

    def plot_losses(self):
        plt.ion()
        fig, axs = plt.subplots(5, 1, figsize=(8, 20))  # 4 subplots (1 for loss, 3 for action components)
        plt.show(block=False)  # Ensure the plot is displayed
        # Add global parameters as text box at the top
        param_text = '\n'.join([f'{key}: {value}' for key, value in self.global_params.items()])
        fig.text(0.5, 0.9, param_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5), ha='center')
        while True:
            try:
                if len(self.critic_losses) > 0:  # Only plot if data is available
                    # Plot Losses (Actor & Critic)
                    axs[0].clear()
                    axs[0].plot(range(len(self.critic_losses)), self.critic_losses, label='Critic Loss')
                    axs[0].plot(range(len(self.actor_losses)), self.actor_losses, label='Actor Loss', color='red')
                    axs[0].set_xlabel('Training Steps')
                    axs[0].set_ylabel('Loss')
                    axs[0].legend()

                    # Ensure we have output tensor values to plot
                    if len(self.output_tensor) > 0:
                        output_array = np.array(self.output_tensor)  # Convert to NumPy array
                        noise_array = np.array(self.output_tensor_noise)

                        for i in range(3):  # x, y, z components
                            axs[i + 1].clear()
                            axs[i + 1].plot(output_array[:, i], label=f'Action {i} (Output)', color='blue')
                            axs[i + 1].plot(noise_array[:, i], label=f'Action {i} (Output + Noise)', linestyle='dashed', color='orange')

                            axs[i + 1].set_xlabel('Steps')
                            axs[i + 1].set_ylabel(f'Velocity {["X", "Y", "Z"][i]}')
                            axs[i + 1].legend()

                    # Plot Reward, DE, and HE Lists in one graph
                    if len(self.reward_list) > 0:
                        min_length = min(len(self.reward_list), len(self.de_list), len(self.he_list))
                        axs[4].clear()
                        axs[4].plot(range(min_length), self.reward_list[:min_length], label='Reward', color='green')
                        axs[4].plot(range(min_length), self.de_list[:min_length], label='DE', color='purple')
                        axs[4].plot(range(min_length), self.he_list[:min_length], label='HE', color='brown')
                        axs[4].set_xlabel('Steps')
                        axs[4].set_ylabel('Values')
                        axs[4].legend()

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                time.sleep(0.2)  # Avoid overloading the CPU
            except Exception as e:
                self.get_logger().error(f"Error in plotting: {e}")
                break  # Exit loop on error

    def reset_uhlennoise(self):
        self.noise_process.reset()
    def state_callback(self, msg):
        #if not self.saved_motions_active:
        # Convert the received data to a tensor
        input_tensor = torch.tensor(msg.data, dtype=torch.float32).view(1, -1)

        normalized_input = self.normalize_input(input_tensor)

        output_tensor = self.actor(normalized_input)[0]
        noise=self.noise_process()

        self.output_tensor.append(output_tensor[0].detach().numpy())

        output_tensor = output_tensor[0]+ noise

        self.output_tensor_noise.append(output_tensor.detach().numpy())

        twist_msg = Twist()
        twist_msg.linear.x = output_tensor[0].item()
        twist_msg.linear.y = output_tensor[1].item()
        twist_msg.angular.z = output_tensor[2].item()
        self.tensor_action_publisher.publish(twist_msg)
        #self.get_logger().info(f'noise: {noise}')
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
        next_state = torch.tensor(msg.next_state, dtype=torch.float32).view(1, -1)

        normalized_state = self.normalize_input(state)
        normalized_next_state = self.normalize_input(next_state)

        self.reward_list.append(reward.item())
        self.de_list.append(normalized_state[0][0].item())
        self.he_list.append(normalized_state[0][2].item())
        #self.get_logger().info(f'normalized_state: {normalized_state}')

        if abs(reward) > 1.0:
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
        #self.get_logger().info(f'Using both buffers: {state_batch} from live + {reward_batch} from long-term')

        # Compute target Q-values
        next_action_batch = self.target_actor(next_state_batch)[0]  # Take action from target actor
        next_state_action_batch = torch.cat([next_state_batch, next_action_batch], dim=1)
        target_value_batch = reward_batch + self.gamma * self.target_critic(next_state_action_batch).detach()
        #target_value_batch = torch.clamp(target_value_batch, -1.0, 1.0)

        # Compute value loss
        predicted_value_batch = self.value(state_batch)
        value_loss = self.value_criterion(predicted_value_batch, target_value_batch)

        # Zero out value gradients before backpropagation
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)  # Clip gradients
        self.value_optimizer.step()

        # Compute advantage
        advantage = target_value_batch - predicted_value_batch.detach()
        #self.get_logger().info(f'Q-Values: {target_value_batch} ')
        # Compute critic loss
        state_action_batch = torch.cat([state_batch, action_batch], dim=1)
        predicted_value_batch = self.critic(state_action_batch)
        critic_loss = self.critic_criterion(predicted_value_batch, target_value_batch)

        # Zero out critic gradients before backpropagation
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # Clip gradients
        self.critic_optimizer.step()

        # Compute actor loss
        predicted_actions = self.actor(state_batch)[0]  # Mean action output
        #actor_loss = -self.critic(torch.cat([state_batch, predicted_actions], dim=1)).mean()
        actor_loss = -(advantage * self.critic(torch.cat([state_batch, predicted_actions], dim=1))).mean()

        # Zero out actor gradients before backpropagation
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # Clip gradients
        self.actor_optimizer.step()

        self.update_target_networks()

        self.get_logger().info(f'-------------------------Computed critic loss: {critic_loss.item()}')
        self.get_logger().info(f'-------------------------Computed actor loss: {actor_loss.item()}')

        # Update target networks

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        current_time = self.get_clock().now().seconds_nanoseconds()[0]  # Get current time in seconds
        self.time_steps.append(current_time)


    def update_target_networks(self):
        tau = self.tau   # Small update factor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
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