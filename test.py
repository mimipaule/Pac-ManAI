import torch
import os

def inspect_model(path):
    print(f"--- Inspecting {path} ---")
    try:
        # Load the file
        data = torch.load(path, map_location='cpu')
        
        # Handle our new dictionary format vs old raw weights format
        if isinstance(data, dict) and 'state_dict' in data:
            weights = data['state_dict']
            print(f"Architecture Type: {data.get('arch', 'unknown')}")
        else:
            weights = data
            print("Architecture Type: Legacy (Raw Weights)")

        # Print shapes of all layers
        for key, value in weights.items():
            # Only print weight layers (skip biases for brevity)
            if 'weight' in key:
                print(f"{key}: {value.shape}")
                
                # Special check for the first Linear layer to explain the 3136 number
                if 'fc.0.weight' in key or 'classifier.0.weight' in key:
                    inputs = value.shape[1]
                    print(f"   -> Flattened Input Size: {inputs}")
                    
    except Exception as e:
        print(f"Error loading model: {e}")
    print("\n")

# List of models to check
models = [
    "pacman_dqn_classic.pt",       # Professor's model
    "pacman_dqn_spiral.pt",        # Your spiral model
    "pacman_dqn_spiral_harder.pt", # Your harder spiral model
    "pacman_dqn_empty.pt"          # Your empty model
]

for m in models:
    if os.path.exists(m):
        inspect_model(m)
    else:
        print(f"File not found: {m}")