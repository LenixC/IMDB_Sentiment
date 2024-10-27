# Sentiment Analysis API with MLOps

A production-ready sentiment analysis API based on IMDB review with monitoring, containerization, and CI/CD pipelines. This project demonstrates MLOps best practices for deploying and maintaining ML models in production.

## Features

- FastAPI-based REST API for sentiment analysis
- Docker containerization
- Prometheus monitoring
- Automated testing with GitHub Actions
- Production-ready project structure

## Quick Start

```bash
# Clone the repository
git clone [your-repo-url]
cd sentiment-analysis-api

# Install dependencies
pip install -r requirements.txt

# Run the API locally
uvicorn main:app --reload --port 8000
```

## API Usage

Send a POST request to analyze sentiment:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was amazing!"}'
```

Expected response:
```json
{
    "text": "This movie was amazing!",
    "sentiment": "POSITIVE",
    "score": 0.9987
}
```

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci_cd.yml
├── Dockerfile
├── requirements.txt
├── main.py
└── test_model.py
```

## Setup Instructions

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn main:app --reload --port 8000
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t sentiment-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 -p 8001:8001 sentiment-api
```

## Monitoring

Prometheus metrics are available at `http://localhost:8001`. Available metrics include:
- Request count
- Response time
- Error rates

## Testing

Run the test suite:
```bash
pytest
```

## Dependencies

- Python 3.9+
- FastAPI
- Transformers
- PyTorch
- Prometheus Client
- Other dependencies listed in requirements.txt

## CI/CD

The project uses GitHub Actions for continuous integration. On each push:
- Runs all tests
- Checks code formatting
- Validates dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

Copyright (c) 2024 Lenix Carter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- HuggingFace Transformers library
- FastAPI framework
- GitHub Actions

