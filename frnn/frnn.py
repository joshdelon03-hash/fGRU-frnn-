# Import the necessary PyTorch libraries
import torch
import torch.nn as nn
import math

class FractionalGRUCell(nn.Module):
    """
    This is a single Fractional Gated Recurrent Unit (fGRU) cell.
    
    It inherits from nn.Module, which is the base class for all neural network
    modules in PyTorch.
    
    The core idea is to modify the standard GRU update equation:
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
    
    To a "fractional" version:
    h_t = (1 - z_t)^alpha * h_{t-1} + (1 - (1 - z_t)^alpha) * h_tilde_t
    
    Here, 'alpha' is a learnable parameter.
    - If alpha = 1.0, this equation simplifies to the standard GRU.
    - If alpha < 1.0, the update is "slower," giving more weight to the previous hidden state (longer memory).
    - If alpha > 1.0, the update is "faster," giving more weight to the new candidate state (shorter memory).
    
    We make 'alpha' a learnable parameter for each hidden unit, so the network
    can learn the optimal memory properties for different features.
    """
    
    def __init__(self, input_size, hidden_size):
        """
        Initializes the layers and parameters for the fGRU cell.
        
        Args:
            input_size (int): The number of expected features in the input 'x'
            hidden_size (int): The number of features in the hidden state 'h'
        """
        super(FractionalGRUCell, self).__init__()
        self.hidden_size = hidden_size
        
        # --- 1. Reset Gate (r_t) ---
        # This gate decides how much of the *past* hidden state (h_{t-1})
        # to forget when calculating the new "candidate" state.
        self.W_r = nn.Linear(input_size, hidden_size, bias=True)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # --- 2. Update Gate (z_t) ---
        # This gate decides how much of the *past* hidden state (h_{t-1})
        # to keep and how much of the *new* candidate state (h_tilde_t) to add.
        self.W_z = nn.Linear(input_size, hidden_size, bias=True)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # --- 3. Candidate Hidden State (h_tilde_t) ---
        # This layer computes the "new" information or content to be added.
        # It uses the reset gate (r_t) to mask the previous hidden state.
        self.W_h = nn.Linear(input_size, hidden_size, bias=True)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # --- 4. The Fractional Order Parameter (alpha) ---
        # This is the core of our "fractional" model.
        # We make it an nn.Parameter so that PyTorch's autograd engine
        # will track its gradients and update it during training.
        # We initialize it with values close to 1.0.
        # We use one 'alpha' per hidden unit for maximum flexibility.
        initial_alpha = torch.ones(hidden_size) + torch.randn(hidden_size) * 0.1
        self.alpha = nn.Parameter(initial_alpha)
        
        # We use Sigmoid activation for the gates (to get values between 0 and 1)
        # and Tanh activation for the candidate state (to get values between -1 and 1).
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_t, h_prev):
        """
        Defines the forward pass for a *single* time step.
        
        Args:
            x_t (torch.Tensor): The input for the current time step.
                                Shape: (batch_size, input_size)
            h_prev (torch.Tensor): The hidden state from the previous time step.
                                   Shape: (batch_size, hidden_size)
                                   
        Returns:
            h_t (torch.Tensor): The new hidden state for the current time step.
                                Shape: (batch_size, hidden_size)
        """
        
        # --- 1. Calculate Reset Gate (r_t) ---
        # r_t = sigma(W_r * x_t + U_r * h_{t-1})
        r_t = self.sigmoid(self.W_r(x_t) + self.U_r(h_prev))
        
        # --- 2. Calculate Update Gate (z_t) ---
        # z_t = sigma(W_z * x_t + U_z * h_{t-1})
        z_t = self.sigmoid(self.W_z(x_t) + self.U_z(h_prev))
        
        # --- 3. Calculate Candidate Hidden State (h_tilde_t) ---
        # h_tilde_t = tanh(W_h * x_t + U_h * (r_t * h_{t-1}))
        # The (r_t * h_prev) part is the element-wise multiplication.
        h_tilde_t = self.tanh(self.W_h(x_t) + self.U_h(r_t * h_prev))
        
        # --- 4. Calculate Final Hidden State (h_t) ---
        # This is where we apply the fractional logic.
        
        # We must ensure alpha is positive, so we can pass it through a
        # softplus function or simply take its absolute value.
        # Using clamp to keep it in a stable range (e.g., 0.1 to 3.0) is also wise.
        # Let's use clamp for stability during training.
        alpha_stable = torch.clamp(self.alpha, 0.1, 3.0)
        
        # The (1 - z_t) term
        one_minus_z = 1.0 - z_t
        
        # The fractional "forget" factor: (1 - z_t)^alpha
        # We add a small epsilon (1e-9) for numerical stability to avoid pow(0, x).
        forget_factor = torch.pow(one_minus_z + 1e-9, alpha_stable)
        
        # The fractional "update" factor: 1 - (1 - z_t)^alpha
        update_factor = 1.0 - forget_factor
        
        # The final fractional update equation:
        # h_t = [ (1 - z_t)^alpha ] * h_{t-1}   +   [ 1 - (1 - z_t)^alpha ] * h_tilde_t
        h_t = forget_factor * h_prev + update_factor * h_tilde_t
        
        return h_t

class FractionalRNN(nn.Module):
    """
    This is the full Fractional RNN model wrapper.
    
    It takes an entire sequence of inputs and iterates the FractionalGRUCell
    over all the time steps. This is the main class you will
    interact with when building a larger model.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the fRNN model.
        
        Args:
            input_size (int): The number of expected features in the input 'x'
            hidden_size (int): The number of features in the hidden state 'h'
            output_size (int): The number of features in the final output
        """
        super(FractionalRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Instantiate our custom fGRU cell
        self.fGRU_cell = FractionalGRUCell(input_size, hidden_size)
        
        # A final fully-connected layer to map the last hidden state
        # to the desired output size (e.g., for classification or regression).
        self.fc_out = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size):
        """
        Generates a zero-initialized hidden state to start the sequence.
        Shape: (batch_size, hidden_size)
        """
        # We use .zeros() to create a tensor of zeros.
        # We would move this to a device (like a GPU) in a real training loop.
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, x):
        """
        Defines the forward pass for an *entire sequence*.
        
        Args:
            x (torch.Tensor): The input sequence.
                              Shape: (batch_size, sequence_length, input_size)
                               
        Returns:
            output (torch.Tensor): The final output from the last time step.
                                   Shape: (batch_size, output_size)
        """
        
        # Get the batch_size and sequence_length from the input tensor
        # x.shape[0] is batch_size
        # x.shape[1] is sequence_length
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Initialize the hidden state
        h_t = self.init_hidden(batch_size)
        
        # We need to loop over the sequence length
        for t in range(seq_len):
            # Get the input for the current time step 't'
            # We use x[:, t, :] which means [all_batches, time_step_t, all_features]
            x_t = x[:, t, :]
            
            # Update the hidden state using our fGRU cell
            # h_t becomes the new h_prev for the next iteration
            h_t = self.fGRU_cell(x_t, h_t)
            
        # After the loop, h_t is the *last* hidden state of the sequence.
        # We can use this last hidden state as a summary of the entire sequence
        # and pass it through our final output layer.
        output = self.fc_out(h_t)
        
        return output