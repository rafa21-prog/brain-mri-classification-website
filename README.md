# NeuroScan - Brain MRI Classification

Academic thesis demonstration website.

## Model
- DenseNet121 feature extractor
- Support Vector Machine (SVM)
- Four classes: Glioma, Healthy, Meningioma, Pituitary
- Explainability: Occlusion Sensitivity

## Project Structure
- app/: Flask application, templates, static assets
- models/: trained model and scaler
- requirements.txt: Python dependencies
- Procfile: production startup command

## Setup
1. Install Python 3.13.
2. Install dependencies:
   pip install -r requirements.txt
3. Set the NEUROSCAN_SECRET_KEY environment variable.
4. Start the application:
   gunicorn --chdir app --bind 0.0.0.0:8000 app:app

## Required Environment Variable
NEUROSCAN_SECRET_KEY=your-long-random-secret

## Important
- Do not commit .env files, user databases, uploaded MRI images,
  prediction history, logs, or backups.
- Configure persistent storage for database and uploaded files
  before production deployment.
- SMS OTP requires a real SMS provider and valid API credentials.
- This is an academic demonstration, not a medical diagnostic system.
