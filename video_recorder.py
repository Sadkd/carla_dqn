import cv2
import os

class VideoRecorder:
    def __init__(self, filename="videos/drive.mp4", fps=20):
        os.makedirs("videos", exist_ok=True)
        self.filename = filename
        self.writer = None
        self.fps = fps

    def add_frame(self, frame):
        if frame is None:
            return

        frame = (frame * 255).astype("uint8")

        if self.writer is None:
            h, w, _ = frame.shape
            self.writer = cv2.VideoWriter(
                self.filename,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (w, h)
            )

        self.writer.write(frame)

    def close(self):
        if self.writer:
            self.writer.release()
