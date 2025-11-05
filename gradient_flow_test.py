import torch
from fGRU_frnn import FractionalRNN

# Setup
INPUT_SIZE = 10
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
BATCH_SIZE = 5
model = FractionalRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
dummy_input = torch.randn(BATCH_SIZE, 20, INPUT_SIZE, requires_grad=True)

# --- The Gradient Test ---
output = model(dummy_input)
# Create a simple, fake loss function (e.g., sum of all output elements)
loss = output.sum() 

# Compute gradients (the backpropagation step)
loss.backward()

# 1. Check a standard Weight Gradient (e.g., the update gate)
# A good model should have gradients flowing back to its weights
w_z_grad = model.fGRU_cell.W_z.weight.grad
print(f"Update Gate Weight Gradient Check (first 5 values): {w_z_grad.flatten()[:5]}")

# 2. Check the Alpha Parameter Gradient (The Custom Part!)
# CRITICAL: This must not be zero or None!
alpha_grad = model.fGRU_cell.alpha.grad

if alpha_grad is not None and not torch.all(alpha_grad == 0):
    print("✅ SUCCESS: Alpha parameter has flowing gradients! The fractional math is working.")
else:
    print("❌ ERROR: Alpha parameter gradient is zero or None. Backprop is broken!")
    
# Clean up for the next test
model.zero_grad()