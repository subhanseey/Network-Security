
#  Network Security ML Deployment System

A production-ready Machine Learning system for network security classification, built with FastAPI and deployed using Docker on Render.

- (https://network-security-kzd9.onrender.com)
-(https://network-security-kzd9.onrender.com/docs)

 Tech Stack
Python, FastAPI, Scikit-learn, Pandas, NumPy, Docker, Render

# Key Features

- End-to-end ML pipeline (data ingestion → training → prediction)
- Trained model and preprocessing artifacts saved for reuse
- REST API for real-time inference using FastAPI
- CSV file upload support for batch prediction
- Fully containerized using Docker for consistent deployment

-# API Endpoints

- `GET /` → Redirects to Swagger UI  
- `POST /predict` → Accepts CSV input and returns predictions  

# Docker Deployment
```bash
docker build -t network-security-app .
docker run -p 8000:8000 network-security-app
