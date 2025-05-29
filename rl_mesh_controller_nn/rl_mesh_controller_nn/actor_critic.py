import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from rl_mesh_controller_msgs.msg import StateActionRewardNextState
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

class ActorNetwork(nn.Module):
    def __init__(self, nodes, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, nodes)
        self.fc2 = nn.Linear(nodes, nodes)
        self.mean = nn.Linear(nodes, action_dim)
        self.log_std = nn.Linear(nodes, action_dim)
        self.log_std.weight.data.uniform_(-3, -2)
        self.log_std.bias.data.uniform_(-3, -2)
        self.log_std.weight.data.fill_(-1)
        self.log_std.bias.data.fill_(-1)

        self.max_action = max_action
        self.min_log_std = -5.
        self.max_log_std = 2.

    def forward(self, state):
        x = torch.tanh(self.fc1(state))  # Tanh activation
        x = torch.tanh(self.fc2(x))
        #mean = torch.tanh(self.mean(x)) * self.max_action  # Squash mean to [-max_action, max_action]
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.min_log_std, self.max_log_std)
        std = log_std.exp()
        mean = torch.tanh(mean) * self.max_action
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        std = std.clamp(min=1e-6)
        print("mean requires grad:", mean.requires_grad, "std requires grad:", std.requires_grad)
        if torch.isnan(mean).any() or torch.isnan(std).any():
            #self.get_logger().error(f"NaN detected in mean or std: mean={mean}, std={std}")
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std) * 0.1

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparametrization trick
        action = torch.tanh(z) * self.max_action

        log_prob = normal.log_prob(z)
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-6))  # Make sure this is differentiable
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

# Define the Q-Network (Critic)
class CriticNetwork(nn.Module):
    def __init__(self, nodes, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, nodes)
        self.fc2 = nn.Linear(nodes, nodes)
        self.fc3 = nn.Linear(nodes, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.tanh(self.fc1(x))  # Tanh activation
        x = torch.tanh(self.fc2(x))  # Tanh activation
        return self.fc3(x)

class SAC(Node):
    def __init__(self, nodes = 128, state_dim=10, action_dim=3, max_action=0.5, lr=1e-3, gamma=0.9, tau=0.005, alpha=.1):
        super().__init__('actor_critic_node')
        self.actor = ActorNetwork(nodes, state_dim, action_dim, max_action)
        self.critic1 = CriticNetwork(nodes,state_dim, action_dim)
        self.critic2 = CriticNetwork(nodes, state_dim, action_dim)
        self.target_critic1 = CriticNetwork(nodes, state_dim, action_dim)
        self.target_critic2 = CriticNetwork(nodes, state_dim, action_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        #self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()#-action_dim  # or another target based on your design
        self.target_entropy = -float(action_dim)
        #self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        #self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))  # Ensure alpha is a trainable parameter

        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha), dtype=torch.float32))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.gamma = gamma
        self.alpha = alpha
        #Noise
        self.tau = tau
        self.sigma = 0.2
        self.theta = 1.

        self.current_episode = []  # Store the current episode's transitions
        self.episode_rewards_list = [] #
        self.episode_rewards = 0   # Track cumulative rewards
        self.episode_count = 0     # Count episodes
        self.episode_succeeded = False # Track episode success

        self.max_episode_length = 200  # Set a max step count per episode
        self.current_step = 0  # Track episode steps

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

        self.terminal_state_subscription = self.create_subscription(Bool,"/terminal_state", self.terminal_state_callback, 10)
        self.terminal_state_subscription # prevent unused variable warning
        self.tensor_action_publisher = self.create_publisher(Twist, '/tensor_action', 10)

        # Initialize the actor and critic networks


        self.noise_process = OrnsteinUhlenbeckNoise(action_dim=3, mu=0, theta=self.theta, sigma=self.sigma)

        self.get_logger().info('ActorCriticNode initialized.')

        # Initialize exploration chance
        self.exploration_chance = 0.0
        self.replay_buffer = ReplayBuffer(capacity=10000)
        #self.replay_buffer_neg = ReplayBuffer(capacity=4096)
        self.long_term_buffer_neg = ReplayBuffer(capacity=10000)
        self.long_term_buffer_pos = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.longterm_batch_size = 500
        #Create timers
        #self.create_timer(2., self.train)
        self.create_timer(30.0, self.reset_uhlennoise)
        # Exploration phase settings
        self.max_exploration_time = 300  # Stop after 5 minutes (300 sec)
        self.exploration_start_time = time.time()
        # Initialize lists to store loss values
        self.actor_losses = []
        self.critic1_losses = []
        self.critic2_losses = []
        self.min_q_losses = []
        self.alpha_entropy = []
        self.time_steps = []
        self.output_tensor = []
        self.output_tensor_noise = []
        self.de_list = []
        self.he_list = []
        self.reward_list = []
        self.global_params = {
            'Learning Rate': lr ,
            'Batch Size': self.batch_size,
            'Gamma': self.gamma,
            'Tau': self.tau,
            'neurons' : nodes,
            }
        # Start the plotting thread
        self.plotting_thread = threading.Thread(target=self.plot_losses)
        self.plotting_thread.start()

    """ state:
    0: de
    1: dee
    2: he
    3,5,7: distance to lookahead
    4,6,8: magnitude to lookahead
    9: angular velocity
    """
    def train(self):
        self.get_logger().info(f'+++++++++++++++++++++++++++++++++++++++******TRAINING******+++++++++++++++++++++++++++++++++++++++')

        # Ensure the replay buffer has enough data before training
        #if replay buffer is empty
        if len(self.replay_buffer) == 0:
            self.get_logger().info(f'Not enough data in replay buffer. Size: {len(self.replay_buffer)}')
            return
        # Sample only from the replay buffer
        sample_size = min(self.batch_size, len(self.replay_buffer.buffer))
        state, action, reward, next_state, done = self.replay_buffer.sample(sample_size)
        #reward = (reward - reward.mean()) / (reward.std() + 1e-6)
        done = self.episode_succeeded
        self.get_logger().info(f'done: {done}')
        # --- Compute target Q-value ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            next_log_prob = next_log_prob.detach()
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            self.get_logger().info(f"next_log_prob requires grad: {next_log_prob.requires_grad}")
            # Compute the entropy loss for automatic tuning
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward + self.gamma * target_q

        new_action, log_prob = self.actor.sample(state)
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        # Update log_alpha via its optimizer
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        self.get_logger().info(f"alpha: {self.alpha}")

        # --- Update Critic Networks ---
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        loss_q1 = F.mse_loss(q1, target_value)
        loss_q2 = F.mse_loss(q2, target_value)

        for _ in range(1):  # Train critic twice for every actor update
            self.critic1_optimizer.zero_grad()
            loss_q1.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            loss_q2.backward()
            self.critic2_optimizer.step()

        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        # --- Update Actor Network ---
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.actor_optimizer.step()

        # --- Soft update target networks ---
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.actor_losses.append(actor_loss.item())
        self.min_q_losses.append(min_q_new.mean().item())
        self.critic1_losses.append(loss_q1.item())
        self.critic2_losses.append(loss_q2.item())
        self.alpha_entropy.append(alpha_loss.item())
        current_time = self.get_clock().now().seconds_nanoseconds()[0]  # Get current time in seconds
        self.time_steps.append(current_time)

        return loss_q1.item(), loss_q2.item(), actor_loss.item()
    def state_buffer_callback(self, msg):
        # Process the StateActionRewardNextState message


        state = torch.tensor(msg.state, dtype=torch.float32, requires_grad=True).view(-1, 10)
        action = torch.tensor(msg.action, dtype=torch.float32).view(-1, 3)
        reward = torch.tensor(msg.reward, dtype=torch.float32).view(-1, 1)
        next_state = torch.tensor(msg.next_state, dtype=torch.float32, requires_grad=True).view(-1, 10)
        done = torch.tensor(msg.is_terminal_state, dtype=torch.float32).view(-1, 1)

        if done.item() == True:
            self.episode_succeeded = True
        #self.get_logger().info("State Buffer Callback")
        #self.get_logger().info(f'State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}')
        normalized_state = self.normalize_input(state)
        normalized_next_state = self.normalize_input(next_state)
        self.episode_rewards += reward.item()
        #self.replay_buffer.push(state, action, reward, next_state, done)
        self.current_episode.append((state, action, reward, next_state, done))

        self.current_step += 1

        self.reward_list.append(self.episode_rewards)
        self.de_list.append(state[0][0].item())
        self.he_list.append(state[0][2].item())
        #self.get_logger().info(f'normalized_state: {normalized_state}')

        #if msg.is_terminal_state:

    def terminal_state_callback(self, msg):
        self.end_episode()
    def end_episode(self):
        for transition in self.current_episode:
            self.replay_buffer.push(*transition)

        self.get_logger().info(f'Episode {self.episode_count} finished with total reward: {self.episode_rewards}')
        self.episode_rewards_list.append((self.episode_count, self.episode_rewards))
        self.train()
        # Reset for the next episode
        self.current_episode = []
        self.episode_rewards = 0
        self.current_step = 0
        self.episode_count += 1
        self.episode_succeeded = False

    def reset_uhlennoise(self):
        self.noise_process.reset()
    def state_callback(self, msg):
        #if not self.saved_motions_active:
        # Convert the received data to a tensor
        input_tensor = torch.tensor(msg.data, dtype=torch.float32).view(1, -1)

        normalized_input = self.normalize_input(input_tensor)
        #self.get_logger().info(f'normalized_input: {normalized_input}')
        #output_tensor, _ = self.actor.sample(normalized_input)
        output_tensor= self.actor(normalized_input)[0]
        noise=self.noise_process()

        self.output_tensor.append(output_tensor[0].detach().numpy())

        output_tensor = output_tensor[0]+ noise

        self.output_tensor_noise.append(output_tensor.detach().numpy())
        #self.get_logger().info(f'output_tensor: {output_tensor}')
        twist_msg = Twist()
        twist_msg.linear.x = output_tensor[0].item()
        twist_msg.linear.y = output_tensor[1].item()
        twist_msg.angular.z = output_tensor[2].item()
        self.tensor_action_publisher.publish(twist_msg)
        #self.get_logger().info(f'noise: {noise}')

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



        # Update target networks


    def save_model(self, file_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            # 'target_actor_state_dict': self.target_actor.state_dict(),  <- Remove or comment this out
            'critic1_state_dict': self.critic1.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }, file_path)
        self.get_logger().info(f'Model saved to {file_path}')

    def load_model(self, file_path):
        try:
            checkpoint = torch.load(file_path, map_location='cpu')
            missing_keys, unexpected_keys = self.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
            self.get_logger().info(f'Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}')
            self.get_logger().info(f'Model loaded from {file_path}')
        except KeyError as e:
            self.get_logger().error(f'Missing key in checkpoint: {e}')
        except FileNotFoundError:
                self.get_logger().warning(f'Checkpoint file not found: {file_path}')
    def plot_losses(self):
            plt.ion()
            fig, axs = plt.subplots(4, 2, figsize=(8, 20))  # 4 subplots (1 for loss, 3 for action components)
            plt.show(block=False)  # Ensure the plot is displayed
            # Add global parameters as text box at the top
            param_text = '\n'.join([f'{key}: {value}' for key, value in self.global_params.items()])
            fig.text(0.5, 0.9, param_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5), ha='center')
            while True:
                try:
                    if len(self.critic1_losses) > 0:  # Only plot if data is available
                        # Plot Losses (Actor & Critic)
                        axs[0,0].clear()
                        axs[0,0].plot(range(len(self.critic1_losses)), self.critic1_losses, label='Critic 1 Loss')
                        axs[0,0].plot(range(len(self.critic1_losses)), self.critic2_losses, label='Critic 2 Loss')
                        axs[0,0].plot(range(len(self.min_q_losses)), self.min_q_losses, label='Min Q')
                        axs[0,0].plot(range(len(self.actor_losses)), self.actor_losses, label='Actor Loss', color='red')
                        axs[0,0].set_xlabel('Training Steps')
                        axs[0,0].set_ylabel('Loss')
                        axs[0,0].legend()

                        # Ensure we have output tensor values to plot
                        if len(self.output_tensor) > 0:
                            output_array = np.array(self.output_tensor)  # Convert to NumPy array
                            noise_array = np.array(self.output_tensor_noise)

                            for i in range(3):  # x, y, z components
                                #row = (i+1) // 2
                                #col = (i+1) % 2
                                axs[i+1, 0].clear()
                                axs[i+1, 0].plot(output_array[:, i], label=f'Action {i} (Output)', color='blue')
                                axs[i+1, 0].plot(noise_array[:, i], label=f'Action {i} (Output + Noise)', linestyle='dashed', color='orange')
                                axs[i+1, 0].set_xlabel('Steps')
                                axs[i+1, 0].set_ylabel(f'Velocity {["X", "Y", "Z"][i]}')
                                axs[i+1, 0].legend()
                        # Plot Episodes total rewars
                        if len(self.episode_rewards_list) > 0:
                            episodes, rewards = zip(*self.episode_rewards_list)
                            axs[0,1].clear()
                            axs[0,1].plot(episodes, rewards, marker='o', linestyle='-', label='Episode Reward')
                            axs[0,1].set_xlabel('Episode')
                            axs[0,1].set_ylabel('Total Reward')
                            axs[0,1].legend()

                        # Plot Reward, DE, and HE Lists in one graph
                        if len(self.reward_list) > 0:
                            min_length = len(self.reward_list)

                            # Plot DE and HE on axs[4]
                            axs[1,1].clear()
                            axs[1,1].plot(range(min(len(self.de_list), min_length)), self.de_list[:min(len(self.de_list), min_length)], label='DE', color='purple')
                            axs[1,1].plot(range(min(len(self.he_list), min_length)), self.he_list[:min(len(self.he_list), min_length)], label='HE', color='brown')
                            axs[1,1].set_xlabel('Steps')
                            axs[1,1].set_ylabel('DE/HE Values')
                            axs[1,1].legend()

                            # Plot Reward on its own axis (e.g., axs[5])
                            axs[2,1].clear()
                            axs[2,1].plot(range(min_length), self.reward_list[:min_length], label='Reward', color='green')
                            axs[2,1].set_xlabel('Steps')
                            axs[2,1].set_ylabel('Reward')
                            axs[2,1].legend()
                        if len(self.alpha_entropy) > 0:
                            axs[3,1].clear()
                            axs[3,1].plot(range(len(self.alpha_entropy)), self.alpha_entropy, label='Alpha Entropy')
                            axs[3,1].set_xlabel('Steps')
                            axs[3,1].set_ylabel('Alpha Entropy')
                            axs[3,1].legend()
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                    time.sleep(0.2)  # Avoid overloading the CPU
                except Exception as e:
                    self.get_logger().error(f"Error in plotting: {e}")
                    break  # Exit loop on error

def main(args=None):
    rclpy.init(args=args)
    node = SAC()
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