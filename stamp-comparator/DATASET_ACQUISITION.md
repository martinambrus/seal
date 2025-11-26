# Dataset Acquisition Guide

Complete guide for building a comprehensive stamp variant training dataset.

## Overview

This guide covers acquiring, validating, and organizing stamp images with documented variants for training your ML models.

---

## Legal and Ethical Considerations

**CRITICAL: Before scraping any website:**

- ✅ **Read Terms of Service** - Understand what's permitted
- ✅ **Check robots.txt** - Respect crawling restrictions
- ✅ **Use Rate Limiting** - Don't overload servers (2-5 seconds between requests)
- ✅ **Consider APIs** - Use official APIs when available
- ✅ **Educational Use** - Ensure your use case is legitimate
- ✅ **Attribution** - Keep records of data sources
- ✅ **Privacy** - Respect personal/copyrighted content

**Recommended approach:** Contact organizations for research data access rather than scraping.

---

## Data Sources

### Tier 1: Official APIs and Partnerships

**Best option for production use:**

1. **Colnect API** (https://colnect.com/en/api)
   - Register for API access
   - Structured data
   - Legal and reliable

2. **Stamps.org Research Access**
   - Contact American Philatelic Society
   - Request research dataset access

3. **Academic Partnerships**
   - Contact university philatelic collections
   - Access digitized archives

### Tier 2: Public Domain Collections

**No restrictions:**

1. **Library of Congress**
   - Digitized stamp collections
   - Public domain images
   - https://www.loc.gov/collections/

2. **Smithsonian Open Access**
   - National Postal Museum collection
   - Free to use
   - https://www.si.edu/openaccess

### Tier 3: Web Scraping (With Permission)

Follow scripts in `backend/scripts/` ONLY after:
- Reading Terms of Service
- Obtaining permission if required
- Implementing proper rate limiting

---

## Workflow

### Step 1: Acquire Base Dataset

```bash
# Option A: Use your existing stamps
python backend/scripts/prepare_training_data.py \
  --source your_stamps/ \
  --output data/training

# Option B: Scrape with permission
python backend/scripts/scrape_colnect.py \
  --output data/scraped_stamps \
  --max-per-term 50 \
  --rate-limit 3.0

# Option C: Download from public domain
# Use provided download scripts for LOC/Smithsonian
```

### Step 2: Pair Variants with Originals

```bash
python backend/scripts/pair_variants_with_originals.py \
  --metadata data/scraped_stamps/metadata.json \
  --output data/training_pairs \
  --validate
```

### Step 3: Manual Review

```bash
# Review pairs visually
# Manually inspect training_pairs.json
# Remove incorrect matches
```

### Step 4: Organize for Training

```bash
# Final organization
python backend/scripts/prepare_training_data.py \
  --source data/training_pairs \
  --output data/final_training \
  --split 0.8
```

---

## Expected Dataset Sizes

### Minimum (Can work with):
- 200-300 stamp images
- 50-100 known variant pairs
- Synthetic variations from training scripts
- **Total**: 500+ training examples

### Recommended:
- 500-1000 stamp images
- 100-200 known variant pairs
- Synthetic variations
- **Total**: 1500+ training examples

### Ideal:
- 2000+ stamp images
- 500+ known variant pairs
- Extensive synthetic variations
- **Total**: 5000+ training examples

---

## Quality vs Quantity

### Quality Markers:
- ✅ High resolution (300+ DPI)
- ✅ Good lighting
- ✅ Clear focus
- ✅ Documented differences
- ✅ Multiple angles/examples

### Priority Order:
1. Real variant pairs with documentation
2. High-quality standard stamps
3. Synthetic variations (generated during training)
4. Lower-quality scraped data

---

## Dataset Organization

```
data/
├── final_training/
│   ├── train/
│   │   ├── stamp_001.jpg
│   │   ├── stamp_002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── stamp_501.jpg
│   │   └── ...
│   └── metadata/
│       ├── training_pairs.json
│       ├── variant_types.json
│       └── sources.json
```

---

## Validation Checklist

Before training:

- [ ] All images load correctly
- [ ] Pairs are properly matched
- [ ] Variant types are documented
- [ ] No duplicate images
- [ ] Balanced variant type distribution
- [ ] Train/val split is appropriate
- [ ] Metadata is complete
- [ ] Sources are documented
- [ ] Legal compliance verified

---

## Troubleshooting

### "Not enough variant pairs"
- Use synthetic variations (generated during training)
- Start with CV methods only
- Gradually collect more data

### "Poor pairing quality"
- Lower similarity threshold
- Manual review and correction
- Use related stamps feature

### "Unbalanced dataset"
- Oversample rare variant types
- Create more synthetic examples
- Use weighted loss functions

---

## Alternative Approaches

If web scraping is not viable:

### 1. Crowdsourcing
- Create submission portal
- Allow collectors to contribute
- Verify submissions manually

### 2. Partnerships
- Contact stamp societies
- Partner with dealers/auctioneers
- Access museum collections

### 3. Purchase
- Buy commercial stamp databases
- License existing datasets
- Commission custom dataset creation

---

## Best Practices

1. **Start Small**: Begin with 200-300 stamps
2. **Iterate**: Train initial models, evaluate, collect more data
3. **Document**: Keep detailed records of sources and methods
4. **Validate**: Manually review critical pairs
5. **Respect**: Always follow legal and ethical guidelines

---

## Resources

- **Colnect**: https://colnect.com/en/stamps
- **StampWorld**: https://www.stampworld.com/
- **APS**: https://stamps.org/
- **Library of Congress**: https://www.loc.gov/collections/
- **Smithsonian**: https://www.si.edu/openaccess

---

## Support

For questions about dataset acquisition:

1. Check legal compliance first
2. Review this guide thoroughly
3. Consider alternative approaches
4. Consult with legal advisors for commercial use

---

## Quick Start Example

```bash
# 1. Prepare your existing stamps
python backend/scripts/prepare_training_data.py \
  --source my_stamps/ \
  --output data/training \
  --variants

# 2. Analyze dataset
python backend/scripts/prepare_training_data.py \
  --source my_stamps/ \
  --analyze

# 3. Train models with your data
python backend/ml_models/train_siamese.py \
  --train_dir data/training/train \
  --val_dir data/training/val

# 4. Evaluate and iterate
```

---

**Remember**: Quality over quantity. A small, well-curated dataset of 500 stamps is better than 5000 poorly labeled images.
