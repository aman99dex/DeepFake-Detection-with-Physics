"""
PhysForensics — Cross-Platform Environment Setup Helper.

Detects your OS + hardware and prints the exact pip install command
for PyTorch, then installs the rest of the requirements.

Usage:
    python setup_env.py           # Auto-detect and install everything
    python setup_env.py --dry-run # Just print commands, don't install
    python setup_env.py --cpu     # Force CPU-only install
"""

import subprocess
import sys
import platform
import argparse


def detect_platform():
    system = platform.system()
    machine = platform.machine()
    python_version = sys.version_info

    info = {
        "os": system,
        "arch": machine,
        "python": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
        "cuda": False,
        "mps": False,
        "cuda_version": None,
    }

    # Check for CUDA
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            info["cuda"] = True
            # Try to get CUDA version
            for line in result.stdout.split("\n"):
                if "CUDA Version:" in line:
                    info["cuda_version"] = line.split("CUDA Version:")[1].strip().split()[0]
    except FileNotFoundError:
        pass

    # Check for Apple Silicon MPS
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        info["mps"] = True

    return info


def get_torch_install_command(info, force_cpu=False):
    """Return the correct pip install command for PyTorch."""
    if force_cpu:
        return (
            'pip install torch torchvision '
            '--index-url https://download.pytorch.org/whl/cpu'
        )

    if info["cuda"]:
        cuda_ver = info.get("cuda_version", "12.1")
        # Map CUDA version to PyTorch wheel
        if cuda_ver and cuda_ver.startswith("11"):
            wheel = "cu118"
        elif cuda_ver and cuda_ver.startswith("12.1"):
            wheel = "cu121"
        else:
            wheel = "cu124"
        return (
            f'pip install torch torchvision '
            f'--index-url https://download.pytorch.org/whl/{wheel}'
        )

    if info["mps"]:
        # Standard PyTorch includes MPS support
        return 'pip install torch torchvision'

    # CPU fallback
    return (
        'pip install torch torchvision '
        '--index-url https://download.pytorch.org/whl/cpu'
    )


def run_command(cmd, dry_run=False):
    print(f"\n  $ {cmd}")
    if not dry_run:
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"  ERROR: command failed with exit code {result.returncode}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only PyTorch install")
    args = parser.parse_args()

    print("=" * 60)
    print("PhysForensics — Environment Setup")
    print("=" * 60)

    info = detect_platform()
    print(f"\nDetected:")
    print(f"  OS:       {info['os']} ({info['arch']})")
    print(f"  Python:   {info['python']}")
    print(f"  CUDA:     {'Yes (' + info['cuda_version'] + ')' if info['cuda'] else 'Not found'}")
    print(f"  Apple MPS: {'Yes (Apple Silicon)' if info['mps'] else 'No'}")

    if info["python"].startswith(("3.8", "3.9")):
        print("\nWARNING: Python 3.10+ is recommended. Some features may not work.")

    # Step 1: PyTorch
    print("\n--- Step 1: Installing PyTorch ---")
    torch_cmd = get_torch_install_command(info, force_cpu=args.cpu)
    run_command(torch_cmd, args.dry_run)

    # Step 2: Core requirements
    print("\n--- Step 2: Installing Core Requirements ---")
    run_command("pip install -r requirements.txt --no-deps torch torchvision 2>/dev/null || "
                "pip install -r requirements.txt", args.dry_run)

    # Step 3: Verify
    print("\n--- Step 3: Verifying Installation ---")
    verify_cmd = (
        'python -c "'
        'import torch; '
        'print(f\'PyTorch {torch.__version__}\'); '
        'print(f\'CUDA: {torch.cuda.is_available()}\'); '
        'print(f\'MPS: {getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()}\'); '
        'import kornia, timm, einops, flask, sklearn; '
        'print(\'All core imports: OK\')"'
    )
    run_command(verify_cmd, args.dry_run)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("\nNext steps:")
    print("  python test_model.py                    # Verify model works")
    print("  python scripts/prepare_data.py          # Download training data")
    print("  python train_v2.py --epochs 5 --synthetic-fallback  # Quick test")
    print("  python app.py                           # Launch dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()
