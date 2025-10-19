import torch
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.interpolate import interp1d
import os
import argparse
from raie_transformer import TrajectoryTransformer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def interpolate_trajectory(frame_ids, coords, target_frame_ids):
    """
    Interpolates missing coordinates for a sequence of frame IDs.
    """
    if len(frame_ids) < 2:
        return frame_ids, coords  # Cannot interpolate with fewer than 2 points

    frame_ids_np = np.array(frame_ids)
    coords_np = np.array(coords)
    target_frame_ids_np = np.array(target_frame_ids)

    interp_x = interp1d(frame_ids_np, coords_np[:, 0], kind='linear', fill_value='extrapolate')
    interp_y = interp1d(frame_ids_np, coords_np[:, 1], kind='linear', fill_value='extrapolate')

    interp_coords = np.vstack([interp_x(target_frame_ids_np), interp_y(target_frame_ids_np)]).T
    return target_frame_ids_np.tolist(), interp_coords.tolist()

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

def automated_trajectory_prediction(video_path, annotations_csv, yolo_model_path, trajectory_model_path, output_dir, rollout_steps=5, min_trajectory_length=10, ball_class=0, delivery_id=None, conf_threshold=0.1):
    """
    Runs YOLO-based ball detection, interpolation, and RAIE Transformer prediction, visualizing results on video.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
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

    # Load YOLO model
    try:
        yolo_model = YOLO(yolo_model_path)
        yolo_model.overrides['conf'] = conf_threshold
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Load trajectory model
    trajectory_model = TrajectoryTransformer(d_model=64, nhead=8, num_encoder_layers=1,
                                            num_decoder_layers=1, dim_feedforward=512, max_len=500).to(device)
    try:
        trajectory_model.load_state_dict(torch.load(trajectory_model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading trajectory model: {e}")
        return
    trajectory_model.eval()

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Process each delivery
    for delivery_id, group in deliveries:
        frame_ids_from_csv = sorted(set(group['frame_id'].values))
        if len(frame_ids_from_csv) < min_trajectory_length + rollout_steps:
            print(f"Skipping delivery {delivery_id}: too few frames ({len(frame_ids_from_csv)})")
            continue

        print(f"Processing delivery {delivery_id}")

        # Run YOLO detection
        trajectory = []
        for frame_id in frame_ids_from_csv:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_id}")
                continue

            results = yolo_model.track(frame, persist=True, classes=[ball_class], verbose=False)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                max_conf_idx = results[0].boxes.conf.argmax().item() if len(results[0].boxes.conf) > 1 else 0
                box = results[0].boxes[max_conf_idx]
                if box.cls.item() == ball_class:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    trajectory.append((frame_id, center_x, center_y))

        # Interpolate missing frames
        if len(trajectory) < len(frame_ids_from_csv):
            detected_frame_ids = [f for f, _, _ in trajectory]
            detected_coords = [[x, y] for _, x, y in trajectory]
            frame_ids_full, coords_full = interpolate_trajectory(detected_frame_ids, detected_coords, frame_ids_from_csv)
        else:
            frame_ids_full = [f for f, _, _ in trajectory]
            coords_full = [[x, y] for _, x, y in trajectory]

        if len(coords_full) < min_trajectory_length + rollout_steps:
            print(f"Skipping delivery {delivery_id}: too few points after interpolation ({len(coords_full)})")
            continue

        # Prepare trajectory for prediction
        coords = torch.tensor(coords_full, dtype=torch.float).to(device)

        # Predict last rollout_steps points
        with torch.no_grad():
            pred_coords = rollout(trajectory_model, coords, steps=rollout_steps)

        # Setup video writer
        output_path = os.path.join(output_dir, f"delivery_{delivery_id}_predictions.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Store points for line trails
        true_points = []
        pred_points = []

        # Rewind video to overlay
        cap = cv2.VideoCapture(video_path)
        for frame_id in frame_ids_from_csv:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            idx = frame_ids_full.index(frame_id) if frame_id in frame_ids_full else None
            if idx is not None:
                x, y = int(coords_full[idx][0]), int(coords_full[idx][1])
                true_points.append((x, y))

                for j in range(1, len(true_points)):
                    cv2.line(frame, true_points[j-1], true_points[j], (255, 128, 128), 5)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            if frame_id in frame_ids_full[-rollout_steps:]:
                pred_idx = frame_ids_full[-rollout_steps:].index(frame_id)
                pred_x, pred_y = int(pred_coords[pred_idx, 0]), int(pred_coords[pred_idx, 1])
                pred_points.append((pred_x, pred_y))

                for j in range(1, len(pred_points)):
                    cv2.line(frame, pred_points[j-1], pred_points[j], (128, 255, 128), 5)
                cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)

            cv2.putText(frame, f"Delivery {delivery_id}, Frame {frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        print(f"Saved video for delivery {delivery_id} to {output_path}")

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated trajectory prediction with YOLO detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--yolo_model", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--trajectory_model", type=str, required=True, help="Path to trajectory model weights")
    parser.add_argument("--output", type=str, required=True, help="Output directory for videos")
    parser.add_argument("--rollout_steps", type=int, default=5, help="Number of rollout steps")
    parser.add_argument("--min_trajectory_length", type=int, default=10, help="Minimum frames for prediction")
    parser.add_argument("--ball_class", type=int, default=0, help="YOLO class ID for ball")
    parser.add_argument("--conf_threshold", type=float, default=0.1, help="YOLO confidence threshold")
    parser.add_argument("--delivery_id", type=int, default=None, help="Specific delivery ID to process")
    args = parser.parse_args()

    automated_trajectory_prediction(
        args.video, args.annotations, args.yolo_model, args.trajectory_model, args.output,
        rollout_steps=args.rollout_steps, min_trajectory_length=args.min_trajectory_length,
        ball_class=args.ball_class, delivery_id=args.delivery_id, conf_threshold=args.conf_threshold
    )
