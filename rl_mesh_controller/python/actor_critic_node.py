import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn

class CustomModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModule, self).__init__()
        self.W = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b = nn.Parameter(torch.randn(hidden_size))
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(torch.addmm(self.b, x, self.W))
        x = torch.tanh(self.linear(x))
        return x

class ActorCriticNode(Node):
    def __init__(self):
        super().__init__('actor_critic_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/model_state',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Initialize the model
        input_size = 12
        hidden_size = 256
        output_size = 3
        self.model = CustomModule(input_size, hidden_size, output_size)
        self.get_logger().info('ActorCriticNode initialized.')

    def listener_callback(self, msg):
        # Convert the received data to a tensor
        input_tensor = torch.tensor(msg.data, dtype=torch.float32).view(1, -1)
        self.get_logger().info(f'Received input tensor: {input_tensor}')

        # Forward pass
        output_tensor = self.model(input_tensor)
        self.get_logger().info(f'Output tensor: {output_tensor}')

def main(args=None):
    rclpy.init(args=args)
    node = ActorCriticNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()