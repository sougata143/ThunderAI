# ThunderAI - Advanced Language Model Management Platform

ThunderAI is a modern platform for managing and deploying large language models (LLMs). It provides an intuitive interface for creating, monitoring, and utilizing various language models, with support for popular architectures like GPT-2, BERT, T5, and LLaMA.

## Features

### 1. Language Model Management
- Support for multiple model architectures:
  - GPT-3.5/4 integration via OpenAI API
  - Custom model support
  - Extensible architecture for new models
- Robust model lifecycle management:
  - Create, read, update, delete operations
  - Model versioning and tracking
  - Automatic error handling and validation
- Secure authentication and authorization
  - JWT-based authentication
  - Bearer token support
  - Role-based access control

### 2. Model Creation and Configuration
- Intuitive model configuration:
  - Model type selection
  - Parameter customization
  - API key management
- Advanced error handling:
  - Detailed error messages
  - Validation feedback
  - Automatic error recovery
- Real-time status updates

### 3. Text Generation
- Interactive text generation interface
- Customizable generation parameters:
  - Maximum tokens control
  - Temperature adjustment
  - Top-p sampling
  - Frequency and presence penalties
- Real-time generation preview
- Generation history tracking

### 4. Security and Performance
- Comprehensive authentication:
  - Secure token management
  - Automatic token refresh
  - Session handling
- Error handling and logging:
  - Detailed error tracking
  - Request/response logging
  - Performance monitoring

## Technology Stack

### Backend
- FastAPI for high-performance API
- MongoDB for flexible data storage
- Motor for async MongoDB operations
- OpenAI API integration
- JWT for authentication

### Frontend
- React 18 with Vite
- Material-UI for modern UI
- Axios for API communication
- Context API for state management
- JWT for secure authentication

## Project Structure

```
ThunderAI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── deps.py     # Dependency injection
│   │   │   └── v1/         # API endpoints
│   │   ├── core/           # Core configuration
│   │   ├── schemas/        # Pydantic models
│   │   ├── services/       # Business logic
│   │   └── main.py        # FastAPI application
│   └── requirements.txt   # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── contexts/      # Auth context
│   │   ├── services/      # API services
│   │   └── App.jsx       # Main application
│   ├── package.json      # Frontend dependencies
│   └── vite.config.js    # Vite configuration
│
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Node.js 16 or higher
- MongoDB 5.0 or higher
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thunderai.git
cd thunderai
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
# backend/.env
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=thunderai
SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_openai_api_key
```

4. Set up the frontend:
```bash
cd frontend
npm install
```

5. Configure frontend environment:
```bash
# frontend/.env
VITE_API_BASE_URL=http://localhost:8001/api/v1
```

### Running the Application

1. Start MongoDB:
```bash
mongod
```

2. Start the backend server:
```bash
cd backend
uvicorn app.main:app --reload --port 8001
```

3. Start the frontend development server:
```bash
cd frontend
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3030
- Backend API: http://localhost:8001
- API Documentation: http://localhost:8001/docs

## Security Considerations

- All API endpoints are protected with JWT authentication
- Tokens are automatically refreshed
- API keys are securely stored
- CORS is properly configured
- Input validation on both frontend and backend

## Error Handling

- Comprehensive error messages
- Automatic error recovery
- Detailed logging
- User-friendly error displays

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.