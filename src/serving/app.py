"""FastAPI application for fraud detection inference."""
from fastapi import FastAPI

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring",
    version="0.1.0",
)

@app.get("/health")
def health():
    return {"status": "ok"}
