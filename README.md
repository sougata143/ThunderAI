# ThunderAI

A powerful machine learning platform for text generation and model experimentation.

## Features

### Core Features
- Advanced Text Generation
  - Support for multiple LLM models
  - Real-time text generation
  - Model-specific parameters configuration
  - Streaming responses

- Model Metrics & Analytics
  - Performance tracking
  - Usage statistics
  - Response time monitoring
  - Model comparison

- Experiments
  - Create and manage ML experiments
  - Track experiment status
  - Model performance metrics
  - Experiment history and versioning

### User Features
- User Authentication
  - Secure JWT-based authentication
  - User registration and login
  - Profile management
  - Role-based access control

- Settings & Configuration
  - Application settings
  - User preferences
  - API key management
  - Model configurations

### Technical Features
- Security
  - JWT token authentication
  - Secure password hashing
  - Protected API endpoints
  - CORS security

- API Integration
  - RESTful API endpoints
  - Swagger documentation
  - API versioning
  - Rate limiting

## Tech Stack

### Frontend
- React with TypeScript
- Material-UI components
- React Router for navigation
- Axios for API requests
- Context API for state management

### Backend
- FastAPI (Python)
- MongoDB for data storage
- JWT for authentication
- OpenAI API integration

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- Python 3.8+
- MongoDB
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thunderai.git
cd thunderai
```

2. Install backend dependencies:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Set up environment variables:
- Create `.env` file in backend directory
```env
SECRET_KEY=your_secret_key
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=thunderai
OPENAI_API_KEY=your_openai_api_key
```

- Create `.env` file in frontend directory
```env
VITE_API_URL=http://localhost:8001
VITE_APP_NAME=ThunderAI
VITE_APP_VERSION=1.0.0
```

5. Start the development servers:

Backend:
```bash
cd backend
uvicorn app.main:app --reload --port 8001
```

Frontend:
```bash
cd frontend
npm run dev
```

## API Documentation

The API documentation is available at `http://localhost:8001/docs` when running the backend server.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for their powerful LLM models
- FastAPI for the excellent Python web framework
- React team for the frontend framework
- All contributors who have helped shape this project