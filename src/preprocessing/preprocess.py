import pandas as pd
import torch
import argparse

def load_data(annotations_csv, output_csv):
    """
    Preprocesses the cricket ball annotations CSV to prepare data for training.
    - Filters out invalid entries and groups by delivery_id.
    - Saves processed sequences to a new CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    annotations = pd.read_csv(annotations_csv)
    annotations = annotations.dropna(subset=['x', 'y'])
    deliveries = annotations.groupby('delivery_id')

    processed_data = []
    for delivery_id, group in deliveries:
        if len(group) < 10:  # Ensure enough frames for sequence
            continue
        x_pixel = torch.tensor(group['x'].values, dtype=torch.float).to(device)
        y_pixel = torch.tensor(group['y'].values, dtype=torch.float).to(device)
        coords = torch.stack([x_pixel, y_pixel], dim=-1)  # (T, 2)
        coords = torch.clamp(coords, min=-1000, max=1000)  # Clip outliers
        for i in range(len(coords)):
            processed_data.append({
                'delivery_id': delivery_id,
                'frame_id': group['frame_id'].iloc[i],
                'x': coords[i, 0].item(),
                'y': coords[i, 1].item(),
                'timestamp': group['timestamp'].iloc[i]
            })

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess cricket ball trajectory data")
    parser.add_argument("--input", type=str, required=True, help="Path to input annotations CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to output processed CSV")
    args = parser.parse_args()
    load_data(args.input, args.output)