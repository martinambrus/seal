# Stamp Comparator

A Python-based application for comparing stamp images using various computer vision and machine learning techniques.

## Project Structure

- **backend/**: Contains the FastAPI application and processing logic.
  - **models/**: Data models.
  - **processors/**: Image processing algorithms (OCR, SSIM, Color, Edge, etc.).
  - **ml_models/**: Machine learning models (Siamese Network, CNN, Autoencoder).
  - **utils/**: Utility functions.
- **frontend/**: React frontend application.
- **data/**: Directory for storing reference and test images.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js and npm (for frontend)

### Backend Setup

1.  Navigate to the project root:
    ```bash
    cd stamp-comparator
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Run the backend server:
    ```bash
    uvicorn backend.main:app --reload
    ```

### Frontend Setup

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```

2.  Install dependencies (you will need to populate package.json first):
    ```bash
    npm install
    ```

3.  Start the development server:
    ```bash
    npm start
    ```
