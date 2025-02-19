import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from rl_mesh_controller_msgs.msg import StateActionRewardNextState
import torch
import torch.nn as nn
import torch.optim as optim

class CustomModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModule, self).__init__()
        self.W = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b = nn.Parameter(torch.randn(hidden_size))
        self.linear = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = torch.relu(torch.addmm(self.b, x, self.W))
        x = torch.tanh(self.linear(x))
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
        self.state_buffer_subscription
        self.tensor_acton_publisher = self.create_publisher(Twist, '/tensor_action', 10)

        # Initialize the model
        input_size = 12
        hidden_size = 256
        output_size = 3
        self.model = CustomModule(input_size, hidden_size, output_size)
        self.target_model = CustomModule(input_size, hidden_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Discount factor for future rewards
        self.gamma = 0.99
        self.get_logger().info('ActorCriticNode initialized.')

    def state_callback(self, msg):
        # Convert the received data to a tensor
        input_tensor = torch.tensor(msg.data, dtype=torch.float32).view(1, -1)
        #self.get_logger().info(f'Received input tensor: {input_tensor}')

        # Forward pass
        output_tensor = self.model(input_tensor)
        self.get_logger().info(f'Output tensor: {output_tensor}')
        # Publish the predicted action as a Twist message
        twist_msg = Twist()
        twist_msg.linear.x = output_tensor[0, 0].item()
        twist_msg.linear.y = output_tensor[0, 1].item()
        twist_msg.angular.z = output_tensor[0, 2].item()
        self.tensor_acton_publisher.publish(twist_msg)
        #self.get_logger().info(f'Published Twist message: {twist_msg}')

    def state_buffer_callback(self, msg):
        state = torch.tensor(msg.state, dtype=torch.float32).view(1, -1)
        action = torch.tensor(msg.action[:3], dtype=torch.float32).view(1, -1)
        reward = torch.tensor([msg.reward], dtype=torch.float32)
        next_state = torch.tensor(msg.next_state, dtype=torch.float32).view(1, -1)

        # self.get_logger().info(f'Received state: {state}')
        # self.get_logger().info(f'Received action: {action}')
        # self.get_logger().info(f'Received reward: {reward}')
        # self.get_logger().info(f'Received next state: {next_state}')

        # Forward pass for the current state
        predicted_action = self.model(state)

        # Forward pass for the next state using the target model
        with torch.no_grad():
            target_action = self.target_model(next_state)

        # Compute the target Q-value
        target_q_value = reward + self.gamma * target_action.max(1)[0]

        # Compute the loss
        loss = self.criterion(predicted_action, action)
        self.get_logger().info(f'Computed loss: {loss.item()}')

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.get_logger().info('Updated model parameters')

        # Update the target model periodically
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': 12,
            'hidden_size': 256,
            'output_size': 3
        }, file_path)
        self.get_logger().info(f'Model saved to {file_path}')



def main(args=None):
    rclpy.init(args=args)
    node = ActorCriticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_model('/home/max/tensor_models/pytorch/model.pth')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()