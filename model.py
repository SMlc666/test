import torch
import torch.nn as nn

class GomokuModel(nn.Module):
    """
    A dual-headed neural network for Gomoku.
    The model takes a board state as input and outputs two things:
    1. Policy: A probability distribution over all possible next moves.
    2. Value: An estimate of the current player's probability of winning.
    """
    def __init__(self, board_size=15, num_input_channels=2, num_resnet_blocks=5, num_filters=128):
        super(GomokuModel, self).__init__()
        self.board_size = board_size

        # Common body
        self.conv_input = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        self.relu_input = nn.ReLU()

        self.resnet_blocks = nn.ModuleList(
            [ResNetBlock(num_filters) for _ in range(num_resnet_blocks)]
        )

        # Policy Head
        self.conv_policy = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.relu_policy = nn.ReLU()
        self.fc_policy = nn.Linear(2 * board_size * board_size, board_size * board_size)
        # Softmax is applied outside the model for the policy head

        # Value Head
        self.conv_value = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.relu_value1 = nn.ReLU()
        self.fc_value1 = nn.Linear(board_size * board_size, 256)
        self.relu_value2 = nn.ReLU()
        self.fc_value2 = nn.Linear(256, 1)
        # Tanh is applied outside the model for the value head

    def forward(self, x):
        # Common body
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = self.relu_input(x)

        for resnet_block in self.resnet_blocks:
            x = resnet_block(x)

        # Policy Head
        policy = self.conv_policy(x)
        policy = self.bn_policy(policy)
        policy = self.relu_policy(policy)
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.fc_policy(policy)

        # Value Head
        value = self.conv_value(x)
        value = self.bn_value(value)
        value = self.relu_value1(value)
        value = value.view(-1, self.board_size * self.board_size)
        value = self.fc_value1(value)
        value = self.relu_value2(value)
        value = self.fc_value2(value)

        return policy, torch.tanh(value)


class ResNetBlock(nn.Module):
    """A standard ResNet block."""
    def __init__(self, num_filters):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual # Skip connection
        out = self.relu2(out)
        return out

if __name__ == '__main__':
    # Example of how to use the model
    board_size = 15
    # Input: Batch of 1, 3 channels (player 1, player 2, current player), 15x15 board
    dummy_input = torch.randn(1, 3, board_size, board_size)
    
    model = GomokuModel(board_size=board_size)
    
    # Get policy and value predictions
    policy_logits, value_estimate = model(dummy_input)
    
    # Apply softmax to get policy probabilities
    policy_probs = torch.softmax(policy_logits, dim=1)
    
    print("Input shape:", dummy_input.shape)
    print("Policy logits shape:", policy_logits.shape)
    print("Policy probabilities shape:", policy_probs.shape)
    print("Value estimate shape:", value_estimate.shape)
    print("\nExample output:")
    print("Value estimate:", value_estimate.item())
    # print("Policy probabilities (sum):", policy_probs.sum().item())
    
    # Verify that the model can be exported to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "gomoku_model_test.onnx",
            input_names=['board_state'],
            output_names=['policy', 'value'],
            dynamic_axes={'board_state': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
        )
        print("\nModel successfully exported to ONNX.")
        import os
        os.remove("gomoku_model_test.onnx")
        print("Cleaned up test ONNX file.")
    except Exception as e:
        print(f"\nError exporting model to ONNX: {e}")


