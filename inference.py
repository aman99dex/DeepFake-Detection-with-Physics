"""
PhysForensics Inference Script.

Detect deepfakes in images or videos.

Usage:
    python inference.py --image path/to/face.jpg
    python inference.py --video path/to/video.mp4
    python inference.py --dir path/to/images/
    python inference.py --checkpoint outputs/checkpoints/best_model.pt --image face.jpg
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from src.models.physforensics import PhysForensics
from src.data.face_processor import FaceProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="PhysForensics Inference")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--video", type=str, help="Path to video")
    parser.add_argument("--dir", type=str, help="Path to directory of images")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    model = PhysForensics(
        image_size=256,
        num_sample_points=1024,
        nerf_hidden=256,
        nerf_layers=8,
        neilf_hidden=128,
        neilf_layers=4,
    ).to(device)

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights")

    model.eval()
    return model


def analyze_image(model, image_path, face_processor, device, threshold=0.5):
    """Analyze a single image for deepfake detection."""
    # Process face
    tensor = face_processor.process_image(image_path).unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        output = model(tensor)

    prob = output["probability"].item()
    anomaly = output["anomaly_score"].item()
    physics = output["physics_scores"].squeeze().cpu().numpy()

    is_fake = prob > threshold
    verdict = "FAKE" if is_fake else "REAL"

    result = {
        "path": str(image_path),
        "verdict": verdict,
        "fake_probability": prob,
        "anomaly_score": anomaly,
        "physics_scores": {
            "energy_conservation": physics[0],
            "specular_consistency": physics[1],
            "brdf_smoothness": physics[2],
            "illumination_coherence": physics[3],
            "temporal_stability": physics[4],
        },
    }
    return result


def print_result(result):
    """Pretty-print analysis result."""
    verdict_color = "\033[91m" if result["verdict"] == "FAKE" else "\033[92m"
    reset = "\033[0m"

    print(f"\n{'='*50}")
    print(f"File: {result['path']}")
    print(f"Verdict: {verdict_color}{result['verdict']}{reset}")
    print(f"Fake Probability: {result['fake_probability']:.4f}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
    print(f"Physics Consistency Scores:")
    for name, score in result["physics_scores"].items():
        bar = "#" * int(score * 50)
        print(f"  {name:<30} {score:.4f} |{bar}")
    print(f"{'='*50}")


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"PhysForensics Deepfake Detector")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    face_processor = FaceProcessor(image_size=256)

    if args.image:
        result = analyze_image(model, args.image, face_processor, device, args.threshold)
        print_result(result)

    elif args.dir:
        dir_path = Path(args.dir)
        results = []
        for img_path in sorted(dir_path.glob("*.*")):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                try:
                    result = analyze_image(model, str(img_path), face_processor, device, args.threshold)
                    results.append(result)
                    print_result(result)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        # Summary
        if results:
            fake_count = sum(1 for r in results if r["verdict"] == "FAKE")
            print(f"\nSummary: {fake_count}/{len(results)} images detected as FAKE")

    elif args.video:
        print("Processing video...")
        frames = face_processor.process_video(args.video, max_frames=32)
        if not frames:
            print("No faces detected in video")
            return

        frame_results = []
        for i, frame_tensor in enumerate(frames):
            with torch.no_grad():
                output = model(frame_tensor.unsqueeze(0).to(device))
            prob = output["probability"].item()
            frame_results.append(prob)

        avg_prob = np.mean(frame_results)
        verdict = "FAKE" if avg_prob > args.threshold else "REAL"
        print(f"\nVideo: {args.video}")
        print(f"Frames analyzed: {len(frame_results)}")
        print(f"Average fake probability: {avg_prob:.4f}")
        print(f"Verdict: {verdict}")

    else:
        print("Please specify --image, --video, or --dir")


if __name__ == "__main__":
    main()
