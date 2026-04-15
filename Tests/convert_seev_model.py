"""
convert_seev_model.py
=====================
Convert SEEV model to TorchScript format compatible with our pipeline.

Their loading:
    loaded = torch.load(model_path)
    trained_state_dict = {f"layers.{key}": value for key, value in loaded.items()}
    model = NNet(architecture)
    model.load_state_dict(trained_state_dict)

Our loading:
    model = torch.jit.load(NN_file, map_location='cpu')
    for _name, param in model.named_parameters(): ...

Usage:
    python convert_seev_model.py --model_path path/to/seev_model.pt \
                                 --system_name hiord8 \
                                 --hidden_layers 2 \
                                 --hidden_size 8 \
                                 --input_dim 8 \
                                 --output_path hiord8_converted.pt
"""

import torch
import torch.nn as nn
import argparse
import numpy as np

# ─────────────────────────────────────────────
# NNet definition — matches SEEV's architecture
# ─────────────────────────────────────────────
class NNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_size, output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# ─────────────────────────────────────────────
# Load SEEV model and convert to TorchScript
# ─────────────────────────────────────────────
def convert(model_path, input_dim, hidden_layers, hidden_size, output_path):
    print(f"Loading SEEV model from {model_path}...")

    loaded = torch.load(model_path, map_location='cpu', weights_only=False)

    # Get raw state dict
    try:
        sd = loaded.state_dict()
        print("Loaded as nn.Module")
    except AttributeError:
        sd = loaded
        print("Loaded as raw state dict")

    print("Raw keys:")
    for k, v in sd.items():
        print(f"  {k}: {v.shape}")

    # Build NNet and load weights
    model = NNet(input_dim, hidden_layers, hidden_size)

    # Remap keys: add 'layers.' prefix if not present
    new_sd = {}
    for k, v in sd.items():
        new_key = k if k.startswith('layers.') else f'layers.{k}'
        new_sd[new_key] = v

    print("\nRemapped keys:")
    for k, v in new_sd.items():
        print(f"  {k}: {v.shape}")

    model.load_state_dict(new_sd, strict=True)
    model = model.double()
    model.eval()
    print("\nModel loaded successfully.")

    # Verify parameters match our pipeline expectations
    print("\nParameters in order (as our pipeline reads them):")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    # Convert to TorchScript
    scripted = torch.jit.script(model)
    scripted.save(output_path)
    print(f"\nSaved TorchScript model to {output_path}")

    # Verify the saved model loads correctly with our pipeline commands
    print("\nVerifying with our pipeline loading commands...")
    loaded_back = torch.jit.load(output_path, map_location=torch.device('cpu'))
    params = []
    for _name, param in loaded_back.named_parameters():
        p = param.detach().numpy()
        params.append(p)
        print(f"  {_name}: {p.shape}")

    print(f"\nTotal parameter arrays: {len(params)}")
    print("Conversion successful!")
    return params

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',    type=str, required=True,
                        help='Path to SEEV .pt model file')
    parser.add_argument('--system_name',   type=str, required=True,
                        help='System name (hiord8, sr, darboux, obs_avoid)')
    parser.add_argument('--hidden_layers', type=int, default=2,
                        help='Number of hidden layers (default 2)')
    parser.add_argument('--hidden_size',   type=int, default=8,
                        help='Hidden layer size (default 8)')
    parser.add_argument('--input_dim',     type=int, default=None,
                        help='Input dimension (inferred from weights if not given)')
    parser.add_argument('--output_path',   type=str, default=None,
                        help='Output path for TorchScript model')
    args = parser.parse_args()

    # Infer input_dim from weights if not provided
    if args.input_dim is None:
        loaded = torch.load(args.model_path, map_location='cpu', weights_only=False)
        try:
            sd = loaded.state_dict()
        except AttributeError:
            sd = loaded
        # First linear layer weight has shape (hidden_size, input_dim)
        first_key = [k for k in sd.keys() if 'weight' in k][0]
        args.input_dim = sd[first_key].shape[1]
        print(f"Inferred input_dim={args.input_dim} from first layer weight")

    # Default output path
    if args.output_path is None:
        args.output_path = f"{args.system_name}_scripted.pt"

    convert(
        args.model_path,
        args.input_dim,
        args.hidden_layers,
        args.hidden_size,
        args.output_path
    )

if __name__ == '__main__':
    main()
