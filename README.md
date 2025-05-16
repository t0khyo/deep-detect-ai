# Deep Detect AI - Graduation Project

## Project Overview
Deep Detect AI is a sophisticated machine learning-based detection system that provides advanced analysis capabilities for various types of data. The project implements a secure, scalable, and maintainable architecture following modern software development practices.

## Academic Information
- **University**: Tanta University
- **Faculty**: Faculty of Computer and Information Technology
- **Supervisor**: Dr. Aida Nasr

## Technical Stack
- **Backend Framework**: Flask
- **Language**: Python 3.x
- **Machine Learning Framework**: PyTorch
- **Key Libraries**:
  - torch & torchvision for deep learning
  - OpenCV for image processing
  - Librosa for audio processing
  - XGBoost for machine learning
  - scikit-learn for additional ML capabilities
  - Pillow for image handling
  - pydub for audio manipulation
- **Containerization**: Docker
- **Production Server**: Waitress

## Project Structure
```
deep-detect-ai/
├── app/           # Main application code
├── model/         # ML models and weights
├── tests/         # Test cases
├── uploads/       # Temporary file storage
├── logs/          # Application logs
├── Dockerfile     # Docker configuration
└── requirements.txt # Python dependencies
```

## Key Features
- Advanced Machine Learning Models
- Image and Audio Processing Capabilities
- RESTful API Architecture
- Containerized Deployment
- Scalable Architecture
- Comprehensive Error Handling
- Production-Ready Configuration

## Prerequisites
- Python 3.x
- Docker and Docker Compose
- CUDA-capable GPU (optional, for faster inference)

## Getting Started

### Local Development Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:

   **Using Flask Development Server:**
   ```bash
   python app/main.py
   ```

   **Using Waitress Production Server:**
   ```bash
   waitress-serve --host=0.0.0.0 --port=5000 app.main:app
   ```
   
   The Waitress server provides better performance and security for local testing. It's recommended to use Waitress when testing production-like scenarios.

### Docker Setup
1. Build and run using Docker Compose:
   ```bash
   docker-compose up --build
   ```

## API Integration
This service is designed to work seamlessly with the Deep Detect Backend (Spring Boot) application. The backend service handles:
- User authentication and authorization
- Request routing and load balancing
- Data persistence
- Cloud storage integration
- API documentation

For detailed backend documentation, please refer to the Deep Detect Backend repository.

## Model Information
The project utilizes state-of-the-art machine learning models for:
- Image analysis and classification
- Audio processing and analysis
- Pattern recognition
- Anomaly detection

## Security Considerations
- Input validation and sanitization
- Secure file handling
- Resource usage monitoring
- Error logging and monitoring

## Acknowledgments
- Dr. Aida Nasr for her guidance and supervision
- Tanta University Faculty of Computer and Information Technology
- All contributors and supporters of this project

## License
This project is part of a graduation thesis and is subject to academic use restrictions. 