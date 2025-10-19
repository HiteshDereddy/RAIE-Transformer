import torch
import pandas as pd
import cv2
import numpy as np
import os
import argparse
from raie_transformer import TrajectoryTransformer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def rollout(model, coords, steps=5):
    """
    Performs autoregressive rollout for trajectory prediction using RAIE Transformer.
    """
    coords_norm = (coords - coords.mean(0)) / coords.std(0).clamp(min=1e-6)
    coords_norm = coords_norm.unsqueeze(0)  # (1, T, 2)

    context_len = coords_norm.size(1) - steps
    src = coords_norm[:, :context_len, :]

    src_T = src.size(1)
    src_pos = torch.arange(src_T, device=src.device, dtype=torch.float) / (src_T - 1 + 1e-6)
    src_pos = src_pos.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    src_aug = torch.cat([src, src_pos], dim=-1)  # (1, T, 3)
    src_pe = torch.arange(src_T, device=src.device).unsqueeze(0)
    memory = model.encoder(model.input_proj(src_aug) + model.pos_encoder(src_pe))

    tgt = coords_norm[:, context_len-1:context_len, :]
    preds = []

    for i in range(steps):
        tgt_T = tgt.size(1)
        tgt_pos = torch.arange(tgt_T, device=tgt.device, dtype=torch.float) / (tgt_T - 1 + 1e-6)
        tgt_pos = tgt_pos.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        tgt_aug = torch.cat([tgt, tgt_pos], dim=-1)  # (1, T, 3)
        tgt_pe = torch.arange(tgt_T, device=src.device).unsqueeze(0)
        out = model.decoder(model.input_proj(tgt_aug) + model.pos_encoder(tgt_pe), memory)
        next_xy = model.pred_head(out[:, -1:, :])
        preds.append(next_xy)
        tgt = torch.cat([tgt, next_xy], dim=1)

    pred_norm = torch.cat(preds, dim=1)  # (1, steps, 2)
    mean = coords.mean(0, keepdim=True)
    std = coords.std(0, keepdim=True).clamp(min=1e-6)
    pred = pred_norm.squeeze(0) * std + mean  # (steps, 2)
    return pred

def visualize_predictions(video_path, annotations_csv, model_path, output_dir, rollout_steps=5, delivery_id=None):
    """
    Visualizes predicted vs. true trajectories on video frames with line trails.
    Saves output videos to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = TrajectoryTransformer(d_model=64, nhead=8, num_encoder_layers=1,
                                 num_decoder_layers=1, dim_feedforward=512, max_len=500).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Ensure the model architecture matches the saved model.")
        return
    model.eval()

    # Load annotations
    try:
        annotations = pd.read_csv(annotations_csv)
        annotations = annotations.dropna(subset=['x', 'y', 'Timestamp'])
        annotations = annotations.rename(columns={'Frame ID': 'frame_id', 'Delivery ID': 'delivery_id'})
        deliveries = annotations.groupby('delivery_id')
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return

    # Get available delivery IDs
    available_delivery_ids = list(deliveries.groups.keys())
    if not available_delivery_ids:
        print("No deliveries found in the CSV.")
        return

    # Filter deliveries
    if delivery_id is not None and delivery_id in available_delivery_ids:
        deliveries = [(delivery_id, deliveries.get_group(delivery_id))]
    elif delivery_id is not None:
        print(f"Delivery ID {delivery_id} not found. Available IDs: {available_delivery_ids}")
        return
    else:
        deliveries = deliveries

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Process each selected delivery
    for delivery_id, group in deliveries:
        if len(group) < rollout_steps + 5:
            print(f"Skipping delivery {delivery_id}: too few frames ({len(group)})")
            continue

        print(f"Processing delivery {delivery_id}")
        coords = torch.tensor(group[['x', 'y']].values, dtype=torch.float).to(device)
        frame_ids = group['frame_id'].values

        # Predict last rollout_steps frames
        with torch.no_grad():
            pred_coords = rollout(model, coords, steps=rollout_steps)

        # Setup video writer
        output_path = os.path.join(output_dir, f"delivery_{delivery_id}_predictions.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Store points for line trails
        true_points = []
        pred_points = []

        # Process frames
        for i, frame_id in enumerate(frame_ids):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Add true point to trail
            x, y = int(coords[i, 0]), int(coords[i, 1])
            true_points.append((x, y))

            # Draw true trajectory (light red line)
            for j in range(1, len(true_points)):
                cv2.line(frame, true_points[j-1], true_points[j], (255, 128, 128), 5)

            # Draw true position (solid red circle)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Draw predicted position and trail for the last rollout_steps frames
            if i >= len(coords) - rollout_steps:
                pred_idx = i - (len(coords) - rollout_steps)
                pred_x, pred_y = int(pred_coords[pred_idx, 0]), int(pred_coords[pred_idx, 1])
                pred_points.append((pred_x, pred_y))

                # Draw predicted trajectory (light green line)
                for j in range(1, len(pred_points)):
                    cv2.line(frame, pred_points[j-1], pred_points[j], (128, 255, 128), 5)

                # Draw predicted position (solid green circle)
                cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)

            # Add text
            cv2.putText(frame, f"Delivery {delivery_id}, Frame {frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert back to BGR for saving
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        print(f"Saved video for delivery {delivery_id} to {output_path}")

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predicted trajectories on video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--output", type=str, required=True, help="Output directory for videos")
    parser.add_argument("--rollout_steps", type=int, default=5, help="Number of rollout steps")
    parser.add_argument("--delivery_id", type=int, default=None, help="Specific delivery ID to process")
    args = parser.parse_args()

    visualize_predictions(args.video, args.annotations, args.model, args.output,
                         rollout_steps=args.rollout_steps, delivery_id=args.delivery_id)