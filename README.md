# frnn: Fractional Recurrent Neural Network

`frnn` is a Python package providing an implementation of a Fractional Gated Recurrent Unit (fGRU) cell and a Fractional Recurrent Neural Network (fRNN) model using PyTorch.

## Features

- **Fractional GRU Cell (`FractionalGRUCell`):** A custom GRU cell that incorporates a learnable fractional order parameter (`alpha`) to control the memory retention properties. This allows the network to learn optimal memory characteristics for different features.
- **Fractional RNN Model (`FractionalRNN`):** A wrapper model that utilizes the `FractionalGRUCell` to process entire sequences, making it easy to integrate into larger deep learning architectures.

## Installation

You can install the `frnn` package directly from its GitHub repository. First, clone the repository:

```bash
git clone https://github.com/yourusername/frnn.git  # Replace with your actual repo URL
cd frnn
```

Then, install it in editable mode (recommended for development):

```bash
pip install -e .
```

## Usage

Here's a basic example of how to use the `FractionalRNN`:

```python
import torch
from frnn import FractionalRNN

# Define model parameters
input_size = 10    # Number of features in each input step
hidden_size = 20   # Number of features in the hidden state
output_size = 5    # Number of features in the output
sequence_length = 7 # Length of the input sequence
batch_size = 3     # Number of samples in a batch

# Create a dummy input tensor
# Shape: (batch_size, sequence_length, input_size)
input_tensor = torch.randn(batch_size, sequence_length, input_size)

# Instantiate the FractionalRNN model
model = FractionalRNN(input_size, hidden_size, output_size)

# Perform a forward pass
output = model(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
print("Model:\n", model)

# You can also access the FractionalGRUCell directly if needed
from frnn import FractionalGRUCell
cell = FractionalGRUCell(input_size, hidden_size)
print("\nFractionalGRUCell:\n", cell)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
