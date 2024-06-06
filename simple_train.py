# Import the torch library, which provides tools for machine learning
import torch

# Import the Jamba model from the jamba.model module
from jamba.model import Jamba

torch.manual_seed(123)

# Create a tensor of random integers between 0 and 100, with shape (1, 100)
# This simulates a batch of tokens that we will pass through the model
x = torch.randint(0, 100, (1, 100))

# Initialize the Jamba model with the specified parameters
# dim: dimensionality of the input data
# depth: number of layers in the model
# num_tokens: number of unique tokens in the input data
# d_state: dimensionality of the hidden state in the model
# d_conv: dimensionality of the convolutional layers in the model
# heads: number of attention heads in the model
# num_experts: number of expert networks in the model
# num_experts_per_token: number of experts used for each token in the input data
model = Jamba(
    dim=512,
    depth=6,
    num_tokens=100,
    d_state=256,
    d_conv=128,
    heads=8,
    num_experts=8,
    num_experts_per_token=2,
)

import time

# Perform a forward pass through the model with the input data
# This will return the model's predictions for each token in the input data
import time
for i in range(2):
    print(f"Warm up iteration {i}")
    start = time.time()
    output = model(x)
    output.sum().backward()
    from torch._dynamo.utils import counters
    dur = time.time() - start
    print(f"Took {dur} seconds")
    print(counters)

print(f"Running actual run")
start = time.time()
output = model(x)
dur = time.time() - start
print(f"Actual run took {dur} seconds")


# Print the model's predictions
print(output)
