# ThunderAI - Advanced Language Model Management Platform

ThunderAI is a modern platform for managing and deploying large language models (LLMs). It provides an intuitive interface for creating, monitoring, and utilizing various language models, with support for popular architectures like GPT-2, BERT, T5, and LLaMA.

## Features

### 1. Language Model Management
- Support for multiple model architectures:
  - GPT-2 for text generation
  - BERT for bidirectional understanding
  - T5 for text-to-text tasks
  - LLaMA for advanced language modeling
- Easy model creation and configuration
- Real-time model status monitoring
- Comprehensive model metrics tracking
- Automatic model versioning

### 2. Model Creation and Configuration
- Simple model initialization
- Customizable training configurations:
  - Batch size optimization
  - Learning rate adjustment
  - Training epochs control
  - Sequence length configuration
- Architecture-specific parameter tuning
- Model status tracking

### 3. Performance Monitoring
- Key metrics tracking:
  - Perplexity scores
  - BLEU scores for translation tasks
  - Model accuracy
  - Training loss
- Real-time performance updates
- Historical performance tracking
- Custom metric visualization

### 4. Text Generation
- Interactive text generation interface
- Customizable generation parameters:
  - Maximum length control
  - Temperature adjustment
  - Top-p sampling
- Real-time generation preview
- Generation history tracking

## Technology Stack

### Backend
- FastAPI for high-performance API
- MongoDB for flexible data storage
- Motor for async MongoDB operations
- PyTorch for model operations
- Transformers library for LLM support

### Frontend
- React with TypeScript
- Material-UI for modern UI components
- React Query for state management
- Axios for API communication
- Chart.js for metrics visualization

## Project Structure

```
ThunderAI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/          # API endpoints
│   │   ├── core/            # Core functionality
│   │   ├── models/          # Data models
│   │   │   └── llm/         # LLM-specific models
│   │   ├── services/        # Business logic
│   │   │   └── llm/         # LLM services
│   │   └── main.py         # FastAPI application
│   └── requirements.txt    # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   │   └── llm/        # LLM-specific components
│   │   ├── services/       # API services
│   │   ├── contexts/       # React contexts
│   │   └── types/         # TypeScript types
│   ├── package.json       # Frontend dependencies
│   └── vite.config.ts     # Vite configuration
│
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Node.js 16 or higher
- MongoDB 5.0 or higher

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thunderai.git
cd thunderai
```

2. Set up the backend:
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn app.main:app --reload --port 8001
```

3. Set up the frontend:
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

4. Access the application:
- Backend API: http://localhost:8001
- Frontend interface: http://localhost:5173

## API Documentation

### Model Management Endpoints

#### Create Model
```http
POST /api/v1/llm/models
```
Request body:
```json
{
  "name": "MyGPT",
  "description": "Custom GPT model for text generation",
  "architecture": "gpt2",
  "training_config": {
    "batch_size": 32,
    "learning_rate": 5e-5,
    "epochs": 3,
    "max_length": 512,
    "model_name": "gpt2"
  }
}
```

#### List Models
```http
GET /api/v1/llm/models
```

#### Get Model
```http
GET /api/v1/llm/models/{model_id}
```

#### Delete Model
```http
DELETE /api/v1/llm/models/{model_id}
```

#### Generate Text
```http
POST /api/v1/llm/models/{model_id}/generate
```
Request body:
```json
{
  "prompt": "Once upon a time",
  "max_length": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.