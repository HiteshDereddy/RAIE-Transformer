import torch
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.interpolate import interp1d
import torch.nn as nn


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# TrajectoryTransformer class
class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=2048, max_len=5000, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoder = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout,
                                                  batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        self.d_model = d_model
        self._max_len = max_len

    @staticmethod
    def _sincos_pos_enc(timestamps, d_model):
        device = timestamps.device
        delta_t = timestamps - timestamps[:, 0:1]
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                             (-np.log(10000.0) / d_model))
        pe = torch.zeros(timestamps.shape + (d_model,), device=device)
        pe[:, :, 0::2] = torch.sin(delta_t.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(delta_t.unsqueeze(-1) * div_term)
        return pe

    def _positional_encoding(self, pos_tensor):
        if pos_tensor is None:
            raise ValueError("pos_tensor required (src_pos / tgt_pos).")
        if pos_tensor.dtype in (torch.int64, torch.long):
            pos_idx = pos_tensor.clamp(0, self._max_len - 1).long()
            return self.pos_encoder(pos_idx)
        else:
            return self._sincos_pos_enc(pos_tensor.float(), self.d_model)

    def forward(self, src, tgt, src_pos, tgt_pos):
        src_emb = self.input_proj(src)
        src_pe = self._positional_encoding(src_pos)
        src_emb = src_emb + src_pe
        src_emb = self.layer_norm(src_emb)
        memory = self.encoder(src_emb)
        tgt_emb = self.input_proj(tgt)
        tgt_pe = self._positional_encoding(tgt_pos)
        tgt_emb = tgt_emb + tgt_pe
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
        dec_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        out = self.pred_head(dec_output)
        return out

# Rollout function
def rollout(model, coords, steps=5):
    coords_norm = (coords - coords.mean(0)) / coords.std(0).clamp(min=1e-6)
    coords_norm = coords_norm.unsqueeze(0)  # (1, T, 2)
    context_len = coords_norm.size(1) - steps
    src = coords_norm[:, :context_len, :]
    src_pos = torch.arange(src.size(1)).unsqueeze(0).to(device)
    memory = model.encoder(model.input_proj(src) + model.pos_encoder(src_pos))
    tgt = coords_norm[:, context_len-1:context_len, :]
    preds = []
    for i in range(steps):
        tgt_pos = torch.arange(tgt.size(1)).unsqueeze(0).to(device)
        out = model.decoder(model.input_proj(tgt) + model.pos_encoder(tgt_pos), memory)
        next_xy = model.pred_head(out[:, -1:, :])
        preds.append(next_xy)
        tgt = torch.cat([tgt, next_xy], dim=1)
    pred_norm = torch.cat(preds, dim=1)  # (1, steps, 2)
    mean = coords.mean(0, keepdim=True)
    std = coords.std(0, keepdim=True).clamp(min=1e-6)
    pred = pred_norm.squeeze(0) * std + mean  # (steps, 2)
    return pred

# Interpolate missing coordinates
def interpolate_trajectory(frame_ids, coords, target_frame_ids):
    if len(frame_ids) < 2:
        return frame_ids, coords  # Cannot interpolate with fewer than 2 points
    
    # Convert to numpy for interpolation
    frame_ids_np = np.array(frame_ids)
    coords_np = np.array(coords)
    target_frame_ids_np = np.array(target_frame_ids)
    
    # Interpolate x and y separately
    interp_x = interp1d(frame_ids_np, coords_np[:, 0], kind='linear', fill_value='extrapolate')
    interp_y = interp1d(frame_ids_np, coords_np[:, 1], kind='linear', fill_value='extrapolate')
    
    # Generate interpolated coordinates for all target frame IDs
    interp_coords = np.vstack([interp_x(target_frame_ids_np), interp_y(target_frame_ids_np)]).T
    return target_frame_ids_np.tolist(), interp_coords.tolist()

# Automated visualization with YOLO detection and interpolation
def automated_trajectory_prediction(video_path, annotations_csv, yolo_model_path, trajectory_model_path, rollout_steps=5, min_trajectory_length=10, ball_class=32, delivery_id=None, conf_threshold=0.1):
    # Load CSV to get frame IDs for selected delivery
    annotations = pd.read_csv(annotations_csv)
    annotations = annotations.dropna(subset=['x', 'y', 'timestamp'])
    deliveries = annotations.groupby('delivery_id')

    # Get available delivery IDs
    available_delivery_ids = list(deliveries.groups.keys())
    if not available_delivery_ids:
        print("No deliveries found in the CSV.")
        return

    # Prompt for delivery ID if not provided
    if delivery_id is None:
        print("Available delivery IDs:", available_delivery_ids)
        delivery_id = input("Enter the delivery ID to visualize (or press Enter to process all): ")
        delivery_id = int(delivery_id) if delivery_id.strip() else None

    # Filter deliveries
    if delivery_id is not None and delivery_id in available_delivery_ids:
        deliveries = [(delivery_id, deliveries.get_group(delivery_id))]
    elif delivery_id is not None:
        print(f"Delivery ID {delivery_id} not found. Available IDs: {available_delivery_ids}")
        return
    else:
        deliveries = deliveries

    # Load YOLO model
    yolo_model = YOLO(yolo_model_path)
    yolo_model.overrides['conf'] = conf_threshold  # Lower confidence for detection

    # Load trajectory model
    trajectory_model = TrajectoryTransformer(d_model=64, num_encoder_layers=1, num_decoder_layers=1).to(device)
    try:
        trajectory_model.load_state_dict(torch.load(trajectory_model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model architecture matches the saved model (check num_encoder_layers and num_decoder_layers).")
        return
    trajectory_model.eval()

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width, frame_height = 1280, 720  # Match annotator resolution
    orig_width, orig_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Process each selected delivery
    for delivery_id, group in deliveries:
        frame_ids_from_csv = sorted(set(group['frame_id'].values))  # Frames to process
        if len(frame_ids_from_csv) < min_trajectory_length + rollout_steps:
            print(f"Skipping delivery {delivery_id}: too few frames ({len(frame_ids_from_csv)})")
            continue

        print(f"Processing delivery {delivery_id}")

        # Run YOLO detection only for specified frame IDs
        trajectory = []
        for frame_id in frame_ids_from_csv:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_id}")
                continue

            # Run YOLO detection (single ball assumption)
            results = yolo_model.track(frame, persist=True, classes=[ball_class], verbose=False)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Take highest-confidence detection
                max_conf_idx = results[0].boxes.conf.argmax().item() if len(results[0].boxes.conf) > 1 else 0
                box = results[0].boxes[max_conf_idx]
                if box.cls.item() == ball_class:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    # Scale to resized resolution
                    center_x = center_x * (1280 / orig_width)
                    center_y = center_y * (720 / orig_height)
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
        output_path = f"delivery_{delivery_id}_predictions.mp4"
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

            # Resize frame
            frame = cv2.resize(frame, (frame_width, frame_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find position (interpolated or detected)
            idx = frame_ids_full.index(frame_id) if frame_id in frame_ids_full else None
            if idx is not None:
                x, y = int(coords_full[idx][0]), int(coords_full[idx][1])
                true_points.append((x, y))

                # Draw true trajectory (light red line)
                for j in range(1, len(true_points)):
                    cv2.line(frame, true_points[j-1], true_points[j], (255, 128, 128), 5)

                # Draw true position (solid red circle)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Draw predicted position and trail for the last rollout_steps frames
            if frame_id in frame_ids_full[-rollout_steps:]:
                pred_idx = frame_ids_full[-rollout_steps:].index(frame_id)
                pred_x, pred_y = int(pred_coords[pred_idx, 0]), int(pred_coords[pred_idx, 1])
                pred_points.append((pred_x, pred_y))

                # Draw predicted trajectory (light green line)
                for j in range(1, len(pred_points)):
                    cv2.line(frame, pred_points[j-1], pred_points[j], (128, 128, 255), 5)

                # Draw predicted position (solid green circle)
                cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)

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
    video_path = "/Users/hitesh/Vertu Live Stream - Yorkshire v Bears - Vitality T20 Blast [CvwVMavj5RM].webm"
    annotations_csv = "/Users/hitesh/Downloads/Complete Traj/smoothed_trajcomp.csv"
    yolo_model_path = "/Users/hitesh/Downloads/content/runs/detect/train3/weights/best.pt"
    trajectory_model_path = "/Users/hitesh/Downloads/aa-2.pth"
    automated_trajectory_prediction(video_path, annotations_csv, yolo_model_path, trajectory_model_path, rollout_steps=5, min_trajectory_length=7, ball_class=0, conf_threshold=0.1)