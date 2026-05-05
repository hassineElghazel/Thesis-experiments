#!/usr/bin/env python3
"""
Modality Gap Experiment — CLI entry point

Usage:
  python run.py synthetic       # Phase 1: 4-config comparison on synthetic data
  python run.py train-vqvae     # Phase 2 prerequisite: train VQ-VAE on Flickr30k
  python run.py flickr          # Phase 2: 5-config comparison on Flickr30k
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "synthetic":
        from experiments.run_synthetic import run_synthetic
        run_synthetic()

    elif cmd == "train-vqvae":
        from experiments.train_vqvae import train_vqvae
        train_vqvae()

    elif cmd == "flickr":
        from experiments.run_flickr import run_flickr
        run_flickr()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
