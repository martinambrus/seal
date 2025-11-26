#!/usr/bin/env python3
"""
Prepare and organize stamp images for ML model training.

This script helps organize 500-1000 stamp images into proper directory
structures for training Siamese networks, CNN detectors, and autoencoders.
"""

import os
import shutil
from pathlib import Path
import random
import argparse
from typing import List, Tuple


def prepare_training_data(
    source_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    create_variants: bool = False
):
    """
    Organize stamp images into training and validation sets.
    
    Args:
        source_dir: Directory containing all stamp images
        output_dir: Output directory for organized data
        train_split: Fraction of data to use for training (rest for validation)
        create_variants: Whether to create synthetic variant examples
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return
    
    # Create output directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(list(source_path.glob(f'*{ext}')))
        all_images.extend(list(source_path.glob(f'*{ext.upper()}')))
    
    if len(all_images) == 0:
        print(f"Error: No images found in '{source_dir}'!")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return
    
    print(f"Found {len(all_images)} images in {source_dir}")
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_split)
    
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"Train set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    # Copy files
    print("\nCopying training images...")
    for i, img_path in enumerate(train_images, 1):
        shutil.copy(img_path, train_dir / img_path.name)
        if i % 100 == 0:
            print(f"  Copied {i}/{len(train_images)} images...")
    
    print("Copying validation images...")
    for i, img_path in enumerate(val_images, 1):
        shutil.copy(img_path, val_dir / img_path.name)
        if i % 100 == 0:
            print(f"  Copied {i}/{len(val_images)} images...")
    
    # Create separate directories for different model types
    if create_variants:
        # For Siamese: create pairs directory
        pairs_dir = output_path / 'pairs'
        pairs_dir.mkdir(exist_ok=True)
        print(f"\nCreated pairs directory at {pairs_dir}")
        
        # For CNN: keep same structure (script will generate variations on-the-fly)
        
        # For Autoencoder: create 'normal' directory
        normal_dir = output_path / 'normal'
        normal_dir.mkdir(exist_ok=True)
        
        # Copy all images to normal directory (assuming all are normal/standard stamps)
        print("Setting up autoencoder data...")
        for i, img_path in enumerate(all_images, 1):
            shutil.copy(img_path, normal_dir / img_path.name)
            if i % 100 == 0:
                print(f"  Copied {i}/{len(all_images)} images...")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")
    
    # Create README with instructions
    readme_path = output_path / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write("Stamp Training Data Structure\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images: {len(all_images)}\n")
        f.write(f"Training images: {len(train_images)}\n")
        f.write(f"Validation images: {len(val_images)}\n\n")
        f.write("Directory Structure:\n")
        f.write(f"- train/ : Training images ({len(train_images)} files)\n")
        f.write(f"- val/ : Validation images ({len(val_images)} files)\n")
        if create_variants:
            f.write(f"- normal/ : All images for autoencoder training\n")
            f.write(f"- pairs/ : Directory for Siamese pair generation\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Training Commands:\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. Siamese Network:\n")
        f.write(f"   python backend/ml_models/train_siamese.py \\\n")
        f.write(f"       --train_dir {train_dir} \\\n")
        f.write(f"       --val_dir {val_dir} \\\n")
        f.write(f"       --epochs 50 --batch_size 16\n\n")
        f.write("2. CNN Detector:\n")
        f.write(f"   python backend/ml_models/train_cnn_detector.py \\\n")
        f.write(f"       --train_dir {train_dir} \\\n")
        f.write(f"       --val_dir {val_dir} \\\n")
        f.write(f"       --epochs 100 --batch_size 8\n\n")
        f.write("3. Autoencoder:\n")
        if create_variants:
            f.write(f"   python backend/ml_models/train_autoencoder.py \\\n")
            f.write(f"       --train_dir {normal_dir} \\\n")
            f.write(f"       --val_dir {val_dir} \\\n")
            f.write(f"       --epochs 200 --batch_size 16\n")
        else:
            f.write(f"   python backend/ml_models/train_autoencoder.py \\\n")
            f.write(f"       --train_dir {train_dir} \\\n")
            f.write(f"       --val_dir {val_dir} \\\n")
            f.write(f"       --epochs 200 --batch_size 16\n")
    
    print(f"\nInstructions saved to {readme_path}")


def analyze_dataset(source_dir: str):
    """
    Analyze the dataset and provide statistics.
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' does not exist!")
        return
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(list(source_path.glob(f'*{ext}')))
        all_images.extend(list(source_path.glob(f'*{ext.upper()}')))
    
    print(f"\nDataset Analysis")
    print("=" * 60)
    print(f"Total images: {len(all_images)}")
    
    if len(all_images) == 0:
        print("No images found!")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return
    
    # Analyze image sizes
    try:
        import cv2
        sizes = []
        
        print("Analyzing image dimensions (sampling first 100 images)...")
        for img_path in all_images[:100]:
            img = cv2.imread(str(img_path))
            if img is not None:
                sizes.append(img.shape[:2])
        
        if sizes:
            avg_height = sum(s[0] for s in sizes) / len(sizes)
            avg_width = sum(s[1] for s in sizes) / len(sizes)
            print(f"\nAverage image size: {int(avg_width)}x{int(avg_height)} (WxH)")
            print(f"Size range: {min(s[1] for s in sizes)}x{min(s[0] for s in sizes)} to "
                  f"{max(s[1] for s in sizes)}x{max(s[0] for s in sizes)}")
    except ImportError:
        print("OpenCV not available, skipping size analysis")
    
    # Recommend training parameters
    print("\n" + "=" * 60)
    print("Recommended Training Parameters:")
    print("=" * 60)
    print(f"- Batch size: {max(4, min(16, len(all_images) // 50))}")
    print(f"- Epochs: {200 if len(all_images) < 1000 else 100}")
    print(f"- Train/Val split: 80/20 ({int(len(all_images)*0.8)}/{int(len(all_images)*0.2)})")
    
    # Training time estimates
    print("\nEstimated Training Time (on GPU):")
    print(f"- Siamese Network: ~{len(all_images) * 0.05:.0f} minutes")
    print(f"- CNN Detector: ~{len(all_images) * 0.1:.0f} minutes")
    print(f"- Autoencoder: ~{len(all_images) * 0.08:.0f} minutes")
    
    print("\nDataset Quality:")
    if len(all_images) < 100:
        print("⚠️  WARNING: Very small dataset (<100 images)")
        print("   Consider collecting more images for better model performance")
    elif len(all_images) < 500:
        print("⚠️  Small dataset (100-500 images)")
        print("   Models may benefit from more training data")
    elif len(all_images) < 1000:
        print("✓  Good dataset size (500-1000 images)")
    else:
        print("✓  Excellent dataset size (>1000 images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare stamp training data for ML models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset
  python prepare_training_data.py --source /path/to/stamps --analyze
  
  # Prepare data with default 80/20 split
  python prepare_training_data.py --source /path/to/stamps --output data/prepared
  
  # Prepare data with custom split and variant directories
  python prepare_training_data.py --source /path/to/stamps --output data/prepared --split 0.85 --variants
        """
    )
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory with stamp images')
    parser.add_argument('--output', type=str,
                       help='Output directory for organized data')
    parser.add_argument('--split', type=float, default=0.8,
                       help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze dataset without copying')
    parser.add_argument('--variants', action='store_true',
                       help='Create variant directories for different models')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.source)
    else:
        if not args.output:
            parser.error("--output is required when not using --analyze")
        
        prepare_training_data(
            args.source,
            args.output,
            args.split,
            args.variants
        )
