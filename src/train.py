import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import yaml
import argparse
from raie_transformer import TrajectoryTransformer

def load_data(annotations_csv):
    """
    Loads and prepares data for training.
    """
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    annotations = pd.read_csv(annotations_csv)
    annotations = annotations.dropna(subset=['x', 'y'])
    deliveries = annotations.groupby('delivery_id')

    data = []
    for _, group in deliveries:
        if len(group) < 10:
            continue
        x_pixel = torch.tensor(group['x'].values, dtype=torch.float).to(device)
        y_pixel = torch.tensor(group['y'].values, dtype=torch.float).to(device)
        coords = torch.stack([x_pixel, y_pixel], dim=-1)
        coords = torch.clamp(coords, min=-1000, max=1000)
        data.append(coords)
    return data

def train_model(model, train_data, epochs=170, lr=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for epoch in range(epochs):
        total_loss = 0
        for coords in train_data:
            if coords.size(0) < 10:
                continue
            coords_norm = (coords - coords.mean(0)) / coords.std(0).clamp(min=1e-6)
            coords_norm = coords_norm.unsqueeze(0)

            context_len = coords_norm.size(1) - 5
            src = coords_norm[:, :context_len, :]
            tgt_inp = coords_norm[:, context_len-1:-1, :]
            tgt_out = coords_norm[:, context_len:, :]

            pred = model(src, tgt_inp)
            loss = torch.mean((pred - tgt_out) ** 2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss={total_loss/len(train_data):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "models/pretrained_model.pth")
    print("Model saved to models/pretrained_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAIE Transformer model")
    parser.add_argument("--data", type=str, required=True, help="Path to processed data CSV")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained model")
    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        config = yaml.safe_load(f)

    model = TrajectoryTransformer(
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 1),
        num_decoder_layers=config.get('num_decoder_layers', 1),
        dim_feedforward=config.get('dim_feedforward', 512),
        max_len=config.get('max_len', 500)
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    data = load_data(args.data)
    train_data, _ = train_test_split(data, test_size=0.2, shuffle=False)
    train_model(model, train_data, epochs=config.get('epochs', 170), lr=config.get('lr', 1e-4))