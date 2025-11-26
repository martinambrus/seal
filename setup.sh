#!/bin/bash

echo "=========================================="
echo "Stamp Comparison Tool - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Found Python $python_version"
else
    echo "Error: Python 3 is not installed!"
    exit 1
fi

# Check Node.js
echo "Checking Node.js version..."
if command -v node &> /dev/null; then
    node_version=$(node --version)
    echo "Found Node.js $node_version"
else
    echo "Error: Node.js is not installed!"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install backend dependencies
echo ""
echo "Installing backend dependencies..."
if [ -f "stamp-comparator/requirements.txt" ]; then
    pip install -r stamp-comparator/requirements.txt
else
    echo "Warning: requirements.txt not found, skipping backend dependencies"
fi

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
if [ -d "stamp-comparator/frontend" ]; then
    cd stamp-comparator/frontend
    npm install
    cd ../..
else
    echo "Warning: frontend directory not found, skipping frontend dependencies"
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p stamp-comparator/data/reference
mkdir -p stamp-comparator/data/test
mkdir -p stamp-comparator/data/variants
mkdir -p stamp-comparator/data/training
mkdir -p stamp-comparator/models/siamese
mkdir -p stamp-comparator/models/cnn_detector
mkdir -p stamp-comparator/models/autoencoder
mkdir -p stamp-comparator/logs

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Place your stamp images in stamp-comparator/data/reference/"
echo ""
echo "2. Prepare training data (if you want to train ML models):"
echo "   python stamp-comparator/backend/scripts/prepare_training_data.py \\"
echo "       --source stamp-comparator/data/reference \\"
echo "       --output stamp-comparator/data/training \\"
echo "       --variants"
echo ""
echo "3. Train models (optional - CV methods work without training):"
echo "   python stamp-comparator/backend/ml_models/train_siamese.py"
echo "   python stamp-comparator/backend/ml_models/train_cnn_detector.py"
echo "   python stamp-comparator/backend/ml_models/train_autoencoder.py"
echo ""
echo "4. Start the backend server:"
echo "   cd stamp-comparator/backend"
echo "   uvicorn main:app --reload --port 8000"
echo ""
echo "5. In a new terminal, start the frontend:"
echo "   cd stamp-comparator/frontend"
echo "   npm run dev"
echo ""
echo "6. Open http://localhost:3000 in your browser"
echo ""
echo "=========================================="
echo ""
echo "For more information, see stamp-comparator/README.md"
echo ""
