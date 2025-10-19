import tkinter as tk
from tkinter import ttk
import cv2
import pandas as pd
from PIL import Image, ImageTk
import numpy as np
from collections import deque

class BallAnnotator:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Enhanced Cricket Ball Annotator")
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.annotations = []
        self.undo_stack = deque(maxlen=100)
        self.redo_stack = deque(maxlen=100)
        self.current_delivery = 1
        self.is_paused = False

        # Get FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            raise ValueError("Could not retrieve FPS. Check video file.")
        print(f"Video FPS: {self.fps}")

        # Main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for video
        self.canvas = tk.Canvas(self.main_frame, width=1280, height=720)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        # Control panel
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X)

        # Frame info
        self.frame_label = tk.Label(self.control_frame, text="Frame: 0 / 0 | Time: 0.0s")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        # Delivery ID
        tk.Label(self.control_frame, text="Delivery ID:").pack(side=tk.LEFT)
        self.delivery_entry = tk.Entry(self.control_frame, width=5)
        self.delivery_entry.insert(0, str(self.current_delivery))
        self.delivery_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Set", command=self.set_delivery).pack(side=tk.LEFT)

        # Camera ID
        tk.Label(self.control_frame, text="Camera ID:").pack(side=tk.LEFT)
        self.camera_id = tk.StringVar(value="Camera 1")
        camera_options = ["Camera 1", "Camera 2"]
        self.camera_menu = ttk.Combobox(self.control_frame, textvariable=self.camera_id, values=camera_options, width=10)
        self.camera_menu.pack(side=tk.LEFT, padx=5)

        # Pitch type
        tk.Label(self.control_frame, text="Pitch Type:").pack(side=tk.LEFT)
        self.pitch_type = tk.StringVar()
        pitch_types = ["", "Fast", "Spin", "Swing", "Bouncer", "Yorker"]
        self.pitch_menu = ttk.Combobox(self.control_frame, textvariable=self.pitch_type, values=pitch_types)
        self.pitch_menu.pack(side=tk.LEFT, padx=5)

        # Ball speed
        tk.Label(self.control_frame, text="Speed (km/h):").pack(side=tk.LEFT)
        self.speed_entry = tk.Entry(self.control_frame, width=5)
        self.speed_entry.pack(side=tk.LEFT, padx=5)

        # Clear Last button (relocated to control frame)
        tk.Button(self.control_frame, text="Clear Last", command=self.clear_last).pack(side=tk.LEFT, padx=5)

        # Scrollbar for frame navigation
        self.scrollbar = tk.Scale(self.main_frame, from_=0, to=self.frame_count-1, orient=tk.HORIZONTAL, 
                                command=self.on_scroll, length=1280, label="Frame")
        self.scrollbar.pack(fill=tk.X, padx=5)

        # Navigation buttons
        nav_frame = tk.Frame(self.main_frame)
        nav_frame.pack(fill=tk.X)
        tk.Button(nav_frame, text="<< 10s", command=self.back_10s).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="<< 10f", command=self.back_10f).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Pause/Play", command=self.toggle_pause).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next Frame", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text=">> 10s", command=self.forward_10s).pack(side=tk.LEFT, padx=5)

        # Action buttons
        action_frame = tk.Frame(self.main_frame)
        action_frame.pack(fill=tk.X)
        tk.Button(action_frame, text="Skip (Invisible)", command=self.skip_frame).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Redo", command=self.redo).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Save", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Load", command=self.load_annotations).pack(side=tk.LEFT, padx=5)

        # Grid toggle
        self.grid_var = tk.BooleanVar()
        tk.Checkbutton(action_frame, text="Show Grid", variable=self.grid_var, command=self.update_frame).pack(side=tk.LEFT, padx=5)

        self.load_frame()
        self.update_frame()

    def load_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1280, 720))
            self.base_frame = frame
            self.update_frame()

    def update_frame(self):
        frame = self.base_frame.copy()
        if self.grid_var.get():
            for x in range(0, 1280, 50):
                cv2.line(frame, (x, 0), (x, 720), (255, 255, 255), 1)
            for y in range(0, 720, 50):
                cv2.line(frame, (0, y), (1280, y), (255, 255, 255), 1)

        self.img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        # Draw existing annotations for current frame
        for ann in self.annotations:
            if ann["frame_id"] == self.current_frame and ann["visibility"] == 1:
                x, y = ann["x"], ann["y"]
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", tags="point")

        # Update frame info and scrollbar
        timestamp = self.current_frame / self.fps
        self.frame_label.config(text=f"Frame: {self.current_frame} / {self.frame_count} | Time: {timestamp:.1f}s")
        self.scrollbar.set(self.current_frame)

    def on_scroll(self, value):
        if self.is_paused:
            return
        self.current_frame = int(float(value))
        self.load_frame()

    def on_click(self, event):
        if self.is_paused:
            return
        x, y = event.x, event.y
        timestamp = self.current_frame / self.fps
        speed = float(self.speed_entry.get()) if self.speed_entry.get() else None
        annotation = {
            "delivery_id": self.current_delivery,
            "frame_id": self.current_frame,
            "x": x,
            "y": y,
            "visibility": 1,
            "timestamp": timestamp,
            "pitch_type": self.pitch_type.get(),
            "ball_speed": speed,
            "camera_id": self.camera_id.get()
        }
        self.annotations.append(annotation)
        self.undo_stack.append(("add", annotation))
        self.redo_stack.clear()
        self.update_frame()

    def skip_frame(self):
        if self.is_paused:
            return
        timestamp = self.current_frame / self.fps
        annotation = {
            "delivery_id": self.current_delivery,
            "frame_id": self.current_frame,
            "x": float("nan"),
            "y": float("nan"),
            "visibility": 0,
            "timestamp": timestamp,
            "pitch_type": self.pitch_type.get(),
            "ball_speed": float(self.speed_entry.get()) if self.speed_entry.get() else None,
            "camera_id": self.camera_id.get()
        }
        self.annotations.append(annotation)
        self.undo_stack.append(("skip", annotation))
        self.redo_stack.clear()
        self.next_frame()

    def clear_last(self):
        if self.is_paused:
            return
        if self.annotations:  # Remove the most recent annotation regardless of frame
            removed = self.annotations.pop()
            self.undo_stack.append(("remove", removed))
            self.redo_stack.clear()
            self.update_frame()

    def undo(self):
        if not self.undo_stack:
            return
        action, annotation = self.undo_stack.pop()
        if action == "add" or action == "skip":
            if self.annotations and self.annotations[-1] == annotation:
                self.annotations.pop()
                self.redo_stack.append((action, annotation))
        elif action == "remove":
            self.annotations.append(annotation)
            self.redo_stack.append((action, annotation))
        self.update_frame()

    def redo(self):
        if not self.redo_stack:
            return
        action, annotation = self.redo_stack.pop()
        if action == "add" or action == "skip":
            self.annotations.append(annotation)
            self.undo_stack.append((action, annotation))
        elif action == "remove":
            if self.annotations and self.annotations[-1] == annotation:
                self.annotations.pop()
                self.undo_stack.append((action, annotation))
        self.update_frame()

    def next_frame(self):
        if self.is_paused:
            return
        self.current_frame += 1
        if self.current_frame < self.frame_count:
            self.load_frame()
        else:
            self.save_annotations()

    def back_10f(self):
        self.current_frame = max(0, self.current_frame - 10)
        self.load_frame()

    def forward_10s(self):
        self.current_frame = min(self.frame_count - 1, self.current_frame + int(10 * self.fps))
        self.load_frame()

    def back_10s(self):
        self.current_frame = max(0, self.current_frame - int(10 * self.fps))
        self.load_frame()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.canvas.config(cursor="wait" if self.is_paused else "")

    def set_delivery(self):
        try:
            self.current_delivery = int(self.delivery_entry.get())
        except ValueError:
            print("Enter a valid number for Delivery ID")

    def save_annotations(self):
        df = pd.DataFrame(self.annotations)
        df.to_csv("annotations.csv", index=False)
        print("Saved to annotations.csv")

    def load_annotations(self):
        try:
            df = pd.read_csv("annotations.csv")
            self.annotations = df.to_dict("records")
            print("Loaded annotations.csv")
            self.update_frame()
        except FileNotFoundError:
            print("No annotations.csv found")

if __name__ == "__main__":
    root = tk.Tk()
    app = BallAnnotator(root, "/Users/hitesh/Vertu Live Stream - Yorkshire v Bears - Vitality T20 Blast [CvwVMavj5RM].webm")
    root.mainloop()