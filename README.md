# ThunderAI

ThunderAI is a comprehensive machine learning platform that provides model training, monitoring, and deployment capabilities with an interactive dashboard for visualization and analysis.

## Features

- 🤖 Multiple ML Model Support (BERT, GPT, LSTM, CNN)
- 📊 Real-time Model Monitoring
- 🔄 Automated Model Retraining
- 📈 Interactive Dashboards
- 🧪 A/B Testing Framework
- 🔍 Advanced Analytics
- 🔐 Authentication & Authorization
- 📦 Docker Support
- ☁️ Cloud Deployment (AWS/GCP)

## Prerequisites

- Python 3.11+
- PostgreSQL
- Redis
- Node.js 16+
- Docker (optional)
- CUDA-compatible GPU (optional)

## Installation

1. Clone the repository:
git clone https://github.com/yourusername/thunderai.git
cd thunderai

2. Set up the Python environment:
### Create and activate virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
### Install dependencies
pip install -r requirements.txt
### Install required language models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

3. Set up the frontend:
cd frontend
npm install

4. Configure environment variables:
### Create .env file in root directory
cp .env.example .env
### Edit .env with your configurations
nano .env


## Database Setup

1. Create PostgreSQL database:
createdb thunderai

2. Run migrations:
alembic upgrade head


## Running the Application

1. Start the backend server:
### From the root directory
uvicorn api.main:app --reload --port 8000

2. Start the frontend development server:
### From the frontend directory
npm start

3. Start Redis server:
redis-server

## Docker Deployment

1. Build and run using Docker Compose:
docker-compose up --build

## Cloud Deployment

### AWS
### Configure AWS credentials
aws configure
### Deploy using Terraform
cd terraform/aws
terraform init
terraform apply

### GCP

### Configure GCP credentials
gcloud auth login
### Deploy using Terraform
cd terraform/gcp
terraform init
terraform apply


## API Documentation

Access the API documentation at:
- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc`

## Monitoring

1. Access Grafana dashboards:
http://localhost:3000

2. View Prometheus metrics:
http://localhost:9090

## Testing

Run the test suite:
### Run all tests
pytest
### Run specific test file
pytest tests/test_models.py
Run with coverage report
pytest --cov=.

## Project Structure
thunderai/
├── api/ # FastAPI application
├── ml/ # Machine learning models
├── monitoring/ # Monitoring configuration
├── frontend/ # React frontend
├── tests/ # Test files
├── docker/ # Docker configuration
├── terraform/ # Infrastructure as Code
└── docs/ # Documentation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, email support@thunderai.com or join our Slack channel.

## Acknowledgments

- Thanks to all contributors
- Built with FastAPI, React, and PyTorch
- Monitoring stack powered by Prometheus and Grafana