#!/usr/bin/env python3
"""
Variant-Original Pairing Script

This script analyzes scraped stamp metadata and attempts to create
training pairs by matching variants with their original versions.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from difflib import SequenceMatcher


class VariantPairer:
    """
    Pairs variant stamps with their original versions for training data.
    """
    
    def __init__(self, metadata_file: str, output_dir: str):
        """
        Args:
            metadata_file: Path to metadata.json from scraping
            output_dir: Where to save training pairs
        """
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.metadata_file, 'r') as f:
            self.stamps = json.load(f)
        
        print(f"Loaded {len(self.stamps)} stamps from metadata")
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def find_potential_pairs(self, similarity_threshold: float = 0.7) -> List[Tuple[Dict, Dict]]:
        """
        Find potential variant-original pairs based on metadata.
        
        Args:
            similarity_threshold: Minimum title similarity to consider a match
        
        Returns:
            List of (variant, original) tuples
        """
        pairs = []
        
        print("Finding potential pairs...")
        
        # Separate variants from potential originals
        variants = []
        potentials = []
        
        for stamp in self.stamps:
            title = stamp.get('title', '').lower()
            desc = stamp.get('description', '').lower()
            
            # Check if it's a known variant
            variant_keywords = ['error', 'variety', 'misprint', 'inverted', 
                              'missing', 'double', 'imperforate']
            is_variant = any(keyword in title or keyword in desc 
                           for keyword in variant_keywords)
            
            if is_variant:
                variants.append(stamp)
            else:
                potentials.append(stamp)
        
        print(f"Found {len(variants)} variants and {len(potentials)} potential originals")
        
        # Try to match variants with originals
        for variant in variants:
            variant_title = variant.get('title', '')
            variant_country = variant.get('country', '')
            variant_year = variant.get('year', '')
            
            # Remove variant keywords from title for matching
            clean_title = variant_title.lower()
            for keyword in ['error', 'variety', 'misprint', 'inverted', 
                          'missing color', 'double print']:
                clean_title = clean_title.replace(keyword, '')
            clean_title = clean_title.strip()
            
            # Find best matching original
            best_match = None
            best_score = 0
            
            for potential in potentials:
                potential_title = potential.get('title', '').lower()
                potential_country = potential.get('country', '')
                potential_year = potential.get('year', '')
                
                # Calculate match score
                title_sim = self.text_similarity(clean_title, potential_title)
                
                # Bonus for matching country and year
                country_match = 1.0 if variant_country == potential_country else 0.5
                year_match = 1.0 if variant_year == potential_year else 0.7
                
                total_score = (title_sim * 0.6 + country_match * 0.2 + year_match * 0.2)
                
                if total_score > best_score:
                    best_score = total_score
                    best_match = potential
            
            if best_match and best_score >= similarity_threshold:
                pairs.append((variant, best_match))
                print(f"Matched: '{variant_title}' with '{best_match['title']}' "
                      f"(score: {best_score:.2f})")
        
        # Also check for related stamps within same entry
        for variant in variants:
            if variant.get('related_local_images'):
                for related_img in variant['related_local_images']:
                    # Treat related images as potential originals
                    original = {
                        'id': f"{variant['id']}_related",
                        'title': f"{variant['title']} (related)",
                        'local_image': related_img['path'],
                        'caption': related_img.get('caption', ''),
                        'related_to': variant['id']
                    }
                    pairs.append((variant, original))
        
        print(f"\nTotal pairs found: {len(pairs)}")
        return pairs
    
    def validate_pair_visually(self, variant: Dict, original: Dict, 
                              min_similarity: float = 0.5) -> bool:
        """
        Validate a pair by checking if images are sufficiently similar.
        
        Args:
            variant: Variant stamp data
            original: Original stamp data
            min_similarity: Minimum SSIM threshold
        
        Returns:
            True if pair appears valid
        """
        variant_img_path = variant.get('local_image')
        original_img_path = original.get('local_image')
        
        if not variant_img_path or not original_img_path:
            return False
        
        try:
            # Load images
            img1 = cv2.imread(variant_img_path)
            img2 = cv2.imread(original_img_path)
            
            if img1 is None or img2 is None:
                return False
            
            # Resize to same size
            target_size = (300, 300)
            img1_resized = cv2.resize(img1, target_size)
            img2_resized = cv2.resize(img2, target_size)
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            
            # Compute SSIM
            from skimage.metrics import structural_similarity as ssim
            similarity = ssim(gray1, gray2)
            
            return similarity >= min_similarity
            
        except Exception as e:
            print(f"Error validating pair: {e}")
            return False
    
    def create_training_pairs(self, pairs: List[Tuple[Dict, Dict]], 
                            validate: bool = True) -> Dict:
        """
        Create organized training pairs dataset.
        
        Args:
            pairs: List of (variant, original) tuples
            validate: Whether to validate pairs visually
        
        Returns:
            Dictionary with training data structure
        """
        valid_pairs = []
        
        print("\nCreating training pairs...")
        
        for idx, (variant, original) in enumerate(pairs, 1):
            print(f"Processing pair {idx}/{len(pairs)}")
            
            # Validate if requested
            if validate:
                if not self.validate_pair_visually(variant, original):
                    print(f"  Skipped: Visual validation failed")
                    continue
            
            # Create pair entry
            pair_id = f"pair_{idx:04d}"
            pair_data = {
                'id': pair_id,
                'variant': {
                    'id': variant['id'],
                    'title': variant.get('title', ''),
                    'image_path': variant.get('local_image', ''),
                    'variant_type': variant.get('variant', {}).get('type', 'unknown'),
                    'description': variant.get('description', '')
                },
                'original': {
                    'id': original['id'],
                    'title': original.get('title', ''),
                    'image_path': original.get('local_image', ''),
                },
                'metadata': {
                    'country': variant.get('country', ''),
                    'year': variant.get('year', ''),
                    'catalog_numbers': variant.get('catalog_numbers', ''),
                    'source': variant.get('source', '')
                }
            }
            
            valid_pairs.append(pair_data)
        
        print(f"\nCreated {len(valid_pairs)} valid training pairs")
        
        # Save pairs
        pairs_file = self.output_dir / 'training_pairs.json'
        with open(pairs_file, 'w') as f:
            json.dump(valid_pairs, f, indent=2)
        
        print(f"Saved to {pairs_file}")
        
        # Create summary
        variant_types = {}
        for pair in valid_pairs:
            vtype = pair['variant']['variant_type']
            variant_types[vtype] = variant_types.get(vtype, 0) + 1
        
        summary = {
            'total_pairs': len(valid_pairs),
            'variant_types': variant_types,
            'sources': list(set(p['metadata']['source'] for p in valid_pairs))
        }
        
        summary_file = self.output_dir / 'dataset_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nDataset Summary:")
        print(f"  Total pairs: {summary['total_pairs']}")
        print(f"  Variant types: {summary['variant_types']}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Pair variant stamps with their original versions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script analyzes scraped stamp metadata and creates training pairs
by matching variants with their original versions using:
- Text similarity matching
- Country and year matching
- Visual similarity validation (optional)

Output: training_pairs.json with structured variant-original pairs
        """
    )
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to metadata.json from scraping')
    parser.add_argument('--output', type=str, default='data/training_pairs',
                       help='Output directory for training pairs')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                       help='Minimum title similarity for matching (0.0-1.0)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate pairs visually using SSIM')
    
    args = parser.parse_args()
    
    # Create pairer
    pairer = VariantPairer(args.metadata, args.output)
    
    # Find potential pairs
    pairs = pairer.find_potential_pairs(args.similarity_threshold)
    
    if not pairs:
        print("\nNo pairs found. Try:")
        print("- Lowering --similarity-threshold")
        print("- Checking your metadata has both variants and originals")
        return
    
    # Create training dataset
    summary = pairer.create_training_pairs(pairs, validate=args.validate)
    
    print("\n" + "=" * 60)
    print("Training pairs ready!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the pairs manually if needed")
    print("2. Use these pairs to augment your training data")
    print("3. Train models with both your original data and these pairs")
    print("\nFiles created:")
    print(f"  - {args.output}/training_pairs.json")
    print(f"  - {args.output}/dataset_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
