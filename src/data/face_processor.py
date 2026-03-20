"""
Face Detection, Alignment, and Preprocessing.

Handles:
- Face detection (MTCNN)
- Face alignment (5-point landmarks)
- Face cropping and normalization
- Video frame extraction
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path


class FaceProcessor:
    """Processes raw images/videos into aligned face crops."""

    def __init__(self, image_size: int = 256, margin: float = 0.3):
        self.image_size = image_size
        self.margin = margin
        self.detector = None

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _init_detector(self):
        """Lazy initialization of face detector."""
        if self.detector is None:
            try:
                from facenet_pytorch import MTCNN
                self.detector = MTCNN(
                    image_size=self.image_size,
                    margin=int(self.image_size * self.margin),
                    keep_all=False,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            except ImportError:
                self.detector = "fallback"

    def detect_and_crop(self, image: Image.Image) -> Image.Image:
        """Detect face and return aligned crop."""
        self._init_detector()

        if self.detector == "fallback":
            # Center crop fallback
            w, h = image.size
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            return image.crop((left, top, left + size, top + size))

        # MTCNN detection
        boxes, probs, landmarks = self.detector.detect(image, landmarks=True)
        if boxes is None or len(boxes) == 0:
            # Fallback to center crop
            w, h = image.size
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            return image.crop((left, top, left + size, top + size))

        # Use highest confidence detection
        box = boxes[0].astype(int)
        x1, y1, x2, y2 = box

        # Add margin
        w, h = x2 - x1, y2 - y1
        margin_x = int(w * self.margin)
        margin_y = int(h * self.margin)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.width, x2 + margin_x)
        y2 = min(image.height, y2 + margin_y)

        return image.crop((x1, y1, x2, y2))

    def process_image(self, image_path: str) -> torch.Tensor:
        """Load, detect face, crop, and transform."""
        image = Image.open(image_path).convert("RGB")
        face = self.detect_and_crop(image)
        return self.transform(face)

    def process_video(self, video_path: str, max_frames: int = 32, stride: int = 5) -> list:
        """Extract and process faces from video frames."""
        try:
            import cv2
        except ImportError:
            return []

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_idx = 0

        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                # Convert BGR to RGB PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                face = self.detect_and_crop(pil_image)
                tensor = self.transform(face)
                frames.append(tensor)

            frame_idx += 1

        cap.release()
        return frames

    def process_directory(self, dir_path: str, output_dir: str = None) -> list:
        """Process all images in a directory."""
        dir_path = Path(dir_path)
        results = []

        for img_path in sorted(dir_path.glob("*.*")):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                try:
                    tensor = self.process_image(str(img_path))
                    results.append({"path": str(img_path), "tensor": tensor})

                    if output_dir:
                        out_path = Path(output_dir) / img_path.name
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        # Save processed face
                        inv_normalize = T.Normalize(
                            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                        )
                        img = inv_normalize(tensor)
                        img = T.ToPILImage()(img.clamp(0, 1))
                        img.save(str(out_path))
                except Exception:
                    continue

        return results
