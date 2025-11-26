# Stamp Comparison Tool

Advanced image analysis application for detecting subtle differences in stamp images using computer vision and machine learning.

![Stamp Comparator](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Node](https://img.shields.io/badge/node-14+-green.svg)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)

## 🎯 Features

### Multiple Detection Methods

- **Computer Vision (No Training Required)**:
  - 📊 **SSIM** - Structural Similarity Index
  - 🔍 **Pixel Diff** - Pixel-level difference detection
  - 🎨 **Color Analysis** - RGB channel comparison
  - 📐 **Edge Detection** - Canny edge-based comparison
  - 📝 **OCR** - Text extraction and matching

- **Machine Learning (Optional)**:
  - 🤖 **Siamese Network** - Deep similarity scoring
  - 🧠 **CNN Detector** - Pixel-wise difference detection
  - 🔬 **Autoencoder** - Anomaly detection

### Interactive UI

- ✅ Side-by-side image comparison
- ✅ Magnifying glass with adjustable zoom
- ✅ Real-time threshold adjustment
- ✅ Multiple visualization modes
- ✅ Color-coded difference overlays
- ✅ Exportable JSON reports

### Flexible Configuration

- ✅ Enable/disable individual methods
- ✅ Adjustable thresholds per method
- ✅ Ensemble fusion (weighted/voting/consensus)
- ✅ Configurable alignment parameters

## 📋 Prerequisites

- **Python** 3.8 or higher
- **Node.js** 14 or higher
- **npm** or yarn
- **(Optional)** CUDA-capable GPU for ML models

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd seal
```

### 2. Run Setup Script

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

The setup script will:
- Create a Python virtual environment
- Install all backend dependencies
- Install all frontend dependencies
- Create necessary directory structure

### 3. Prepare Your Data

Place your stamp images in `stamp-comparator/data/reference/`:

```bash
cp /path/to/your/stamps/* stamp-comparator/data/reference/
```

### 4. Start the Application

**Terminal 1 - Backend:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd stamp-comparator/backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd stamp-comparator/frontend
npm run dev
```

### 5. Open Application

Navigate to **`http://localhost:3000`** in your web browser.

## 📖 Usage

### Basic Comparison (No Training Required)

The **5 computer vision methods** work immediately without any training:

1. Upload a **reference stamp** image
2. Upload a **test stamp** image
3. Select detection methods (SSIM, Pixel Diff, Color, Edge, OCR)
4. Click **"Compare Stamps"**
5. Review results and adjust thresholds as needed

### Training ML Models (Optional)

For enhanced accuracy with **Siamese Network**, **CNN Detector**, and **Autoencoder**:

#### Step 1: Prepare Training Data

```bash
python stamp-comparator/backend/scripts/prepare_training_data.py \
  --source stamp-comparator/data/reference \
  --output stamp-comparator/data/training \
  --variants \
  --split 0.8
```

This will:
- Split images into 80% training, 20% validation
- Create organized directory structure
- Generate a README with training commands

#### Step 2: Train Models

**Siamese Network** (Similarity Scoring):
```bash
python stamp-comparator/backend/ml_models/train_siamese.py \
  --train_dir stamp-comparator/data/training/train \
  --val_dir stamp-comparator/data/training/val \
  --epochs 50 \
  --batch_size 16
```

**CNN Detector** (Pixel-wise Differences):
```bash
python stamp-comparator/backend/ml_models/train_cnn_detector.py \
  --train_dir stamp-comparator/data/training/train \
  --val_dir stamp-comparator/data/training/val \
  --epochs 100 \
  --batch_size 8
```

**Autoencoder** (Anomaly Detection):
```bash
python stamp-comparator/backend/ml_models/train_autoencoder.py \
  --train_dir stamp-comparator/data/training/normal \
  --val_dir stamp-comparator/data/training/val \
  --model_type standard \
  --epochs 200 \
  --batch_size 16
```

**Training Times** (approximate, 500-1000 images):
- Siamese Network: 2-4 hours
- CNN Detector: 4-6 hours
- Autoencoder: 3-5 hours

*⚡ GPU highly recommended for training (10-20x faster)*

## ⚙️ Configuration

### Method Parameters

Each detection method has configurable thresholds:

| Method | Default Threshold | Description |
|--------|-------------------|-------------|
| **SSIM** | 0.95 | Structural similarity (higher = more similar) |
| **Pixel Diff** | 0.20 | Pixel-level differences (lower = more strict) |
| **Color** | 0.25 | RGB channel differences |
| **Edge** | 0.15 | Edge detection differences |
| **OCR** | 0.80 | Text matching confidence |
| **Siamese** | 0.90 | Neural similarity threshold |
| **CNN** | 0.75 | CNN detection confidence |
| **Autoencoder** | 0.30 | Reconstruction error threshold |

### Alignment Settings

- **Method**: ORB (default), SIFT
- **Min Quality**: 0.7 (minimum alignment score to proceed)
- **Feature Count**: 5000 (number of features to detect)

### Ensemble Fusion

- **Weighted**: Combines methods using confidence-weighted averaging
- **Voting**: Majority voting across methods
- **Consensus**: Requires agreement from multiple methods

## 📁 Project Structure

```
seal/
├── stamp-comparator/
│   ├── backend/
│   │   ├── main.py                 # FastAPI application
│   │   ├── config.py               # Configuration management
│   │   ├── models/                 # Data models
│   │   │   └── comparison_result.py
│   │   ├── processors/             # CV analysis methods
│   │   │   ├── alignment.py
│   │   │   ├── ssim_analyzer.py
│   │   │   ├── pixel_diff_analyzer.py
│   │   │   ├── color_analyzer.py
│   │   │   ├── edge_analyzer.py
│   │   │   └── ocr_processor.py
│   │   ├── ml_models/              # ML implementations
│   │   │   ├── siamese_network.py
│   │   │   ├── cnn_detector.py
│   │   │   ├── autoencoder.py
│   │   │   ├── train_*.py          # Training scripts
│   │   │   └── *_inference.py      # Inference wrappers
│   │   ├── utils/                  # Utilities
│   │   │   ├── fusion.py
│   │   │   └── image_utils.py
│   │   └── scripts/
│   │       └── prepare_training_data.py
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── components/         # React components
│   │   │   │   ├── ImageUploader.jsx
│   │   │   │   ├── ImageViewer.jsx
│   │   │   │   ├── MagnifyingGlass.jsx
│   │   │   │   ├── MethodControls.jsx
│   │   │   │   ├── ViewModeSelector.jsx
│   │   │   │   └── ResultsPanel.jsx
│   │   │   ├── services/
│   │   │   │   └── api.js
│   │   │   ├── App.jsx
│   │   │   └── App.css
│   │   ├── package.json
│   │   └── vite.config.js
│   ├── data/
│   │   ├── reference/              # Reference images
│   │   ├── test/                   # Test images
│   │   └── training/               # Training data (generated)
│   ├── models/                     # Trained model checkpoints
│   │   ├── siamese/
│   │   ├── cnn_detector/
│   │   └── autoencoder/
│   ├── requirements.txt
│   └── README.md
├── setup.sh                        # Linux/Mac setup
├── setup.bat                       # Windows setup
└── README.md                       # This file
```

## 🔧 Troubleshooting

### Common Issues

**"Alignment failed" error:**
- Ensure both images contain the same stamp type
- Try increasing feature count in alignment settings
- Check image quality (avoid blurry or dark images)
- Verify images are properly oriented

**OCR not detecting text:**
- Install Tesseract OCR:
  - **Ubuntu**: `sudo apt-get install tesseract-ocr`
  - **Mac**: `brew install tesseract`
  - **Windows**: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

**ML models not loading:**
- Train the models first (see Training ML Models section)
- Verify model files exist in `models/` directories
- Check console output for specific error messages
- Ensure PyTorch is installed correctly

**Frontend not connecting to backend:**
- Ensure backend is running on port 8000
- Check CORS settings in `backend/main.py`
- Verify proxy configuration in `frontend/vite.config.js`
- Check browser console for errors

### Performance Tips

- 🚀 Use GPU for ML models (10-20x faster)
- 📏 Resize large images before comparison (recommended: 1000x1000px)
- ⚡ Disable unused detection methods
- 🎯 Start with CV methods, add ML models once trained
- 💾 Use SSD for faster data loading during training

## 📚 API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

- `POST /api/compare` - Compare two stamp images
- `GET /api/config/default` - Get default configuration
- `POST /api/config/save` - Save custom configuration
- `GET /api/config/load/{name}` - Load saved configuration
- `GET /health` - Health check

## 🛠️ Development

### Adding New Detection Methods

1. Create analyzer in `backend/processors/`
2. Implement `analyze()` method returning `MethodResult`
3. Add to `backend/main.py` comparison pipeline
4. Update `frontend/src/components/MethodControls.jsx`
5. Add configuration in `backend/config.py`

### Running Tests

```bash
# Backend tests
cd stamp-comparator/backend
pytest

# Frontend tests
cd stamp-comparator/frontend
npm test
```

### Code Style

- **Backend**: Black formatter, flake8 linter
- **Frontend**: ESLint, Prettier

## 📄 License

GNU AFFERO GENERAL PUBLIC LICENSE Version 3

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Computer vision algorithms based on OpenCV
- ML architectures inspired by research in image similarity and anomaly detection
- Built with **FastAPI**, **React**, **OpenCV**, and **PyTorch**
- UI design inspired by modern web applications

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Made with ❤️ for stamp collectors and philatelists worldwide**
