# Automatic SQL Injection Vulnerability Scanner

An automatic vulnerability scanning tool for detecting SQL Injection (SQLi)
attacks using a machine-learning based approach.

The project trains a supervised ML model on labelled SQL query payloads, then
exposes the trained detector through both a command-line prediction script and
a FastAPI service. The API can classify individual query strings, scan batches,
or extract and scan parameters from a simulated HTTP request.

## Project Goals

- Detect SQL Injection payloads before they reach application database logic.
- Combine text features with handcrafted security features such as SQL keyword
  counts, special-character patterns, entropy, and obfuscation signals.
- Provide a reusable API that can sit in front of web applications as a scanning
  layer.
- Support retraining and evaluation from raw labelled data.

## Project Structure

```text
sqli_detector/
|-- api/                  # FastAPI app, request schemas, scan helpers
|-- data/
|   |-- raw/              # Original labelled dataset
|   `-- processed/        # Generated cleaned splits and feature matrices
|-- models/               # Generated trained model and TF-IDF vectorizer
|-- notebooks/            # Exploration, feature engineering, training notes
|-- reports/              # Generated metrics and evaluation figures
|-- src/                  # Data cleaning, feature engineering, training, inference
|-- requirements.txt      # Python dependencies
`-- README.md
```

## ML Pipeline

The pipeline is split into four main stages:

1. **Data cleaning**: reads `data/raw/trainingdata.csv`, removes duplicates and
   invalid rows, normalizes query strings, and creates train/validation/test
   splits.
2. **Feature engineering**: builds TF-IDF features and handcrafted SQLi signals.
3. **Model training**: compares candidate classifiers and saves the best model.
4. **Evaluation/inference**: reports model performance and serves predictions.

The current saved model is a calibrated Linear SVM. The default prediction
threshold is `0.35`, which is intentionally lower than `0.5` to reduce missed
attacks in production-style traffic where most requests are benign.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If PowerShell blocks activation scripts, run PowerShell as your user and allow
local scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Rebuild the Pipeline

Run these commands from the project root.

Clean and split the raw data:

```powershell
python src/data_cleaning.py
```

Build feature matrices and the TF-IDF vectorizer:

```powershell
python src/features.py
```

Train and select the best model:

```powershell
python src/train.py
```

Evaluate the model on the test set:

```powershell
python src/evaluate.py
```

Generated outputs are written to `data/processed/`, `models/`, and `reports/`.
These folders contain reproducible artifacts and are ignored by Git.

## Command-Line Prediction

Classify one query:

```powershell
python src/predict.py --query "SELECT * FROM users WHERE id=1 OR 1=1--"
```

Classify queries from a text file:

```powershell
python src/predict.py --file queries.txt
```

Override the detection threshold:

```powershell
python src/predict.py --query "' OR 1=1--" --threshold 0.45
```

## Run the API

Start the FastAPI service:

```powershell
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

- `GET /health`: checks whether the model is loaded.
- `POST /detect`: classifies one query string.
- `POST /detect/batch`: classifies multiple query strings.
- `POST /scan`: extracts request parameters and scans each value.

Example single-query detection request:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/detect" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"SELECT * FROM users WHERE id=1 OR 1=1--","threshold":0.35}'
```

Example request scan:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/scan" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"url":"/login?id=1 OR 1=1--","form_data":{"username":"admin"},"threshold":0.35}'
```

## Use as Active Middleware

The scanner can run inside a Python web application and block suspicious
requests before route/database code runs.

FastAPI or Starlette:

```python
from fastapi import FastAPI

from api.middleware import SQLiDetectionMiddleware
from src.predict import load_pipeline

pipeline = load_pipeline()
app = FastAPI()

app.add_middleware(
    SQLiDetectionMiddleware,
    pipeline=pipeline,
    threshold=0.35,
    skip_paths={"/health", "/docs", "/openapi.json", "/redoc"},
)
```

Flask:

```python
from flask import Flask

from api.middleware import register_flask_middleware
from src.predict import load_pipeline

app = Flask(__name__)
pipeline = load_pipeline()

register_flask_middleware(app, pipeline, threshold=0.35)
```

Django:

```python
MIDDLEWARE = [
    "api.middleware.DjangoSQLiMiddleware",
    # ...
]

SQLI_THRESHOLD = 0.35
SQLI_SKIP_PATHS = {"/health", "/admin/login/"}
```

You can also set the threshold with an environment variable:

```powershell
$env:SQLI_THRESHOLD = "0.35"
```

The FastAPI/Starlette middleware safely replays the request body after scanning,
so normal route handlers can still read JSON or form data.

## Current Model Performance

The latest validation report selected Linear SVM as the best model:

- Macro F1: `0.9961`
- ROC AUC: `0.9997`
- Benign recall: `0.9976`
- SQLi recall: `0.9950`

Metrics and figures are generated under `reports/`.

## Development Notes

- Keep `venv/`, `__pycache__/`, `data/processed/`, `models/*.pkl`, and
  `reports/` out of version control.
- `data/raw/trainingdata.csv` is the source dataset for rebuilding the pipeline.
- `api/schema.py` contains FastAPI request/response models.
- `api/middleware.py` contains parameter extraction and scanning helpers.
- `src/predict.py` is shared by the CLI and API for inference.

## Security Note

This tool is a detection layer, not a replacement for secure coding practices.
Applications should still use parameterized queries, ORM protections, input
validation, least-privilege database users, logging, and monitoring. The ML
scanner should be treated as an additional defensive control.
