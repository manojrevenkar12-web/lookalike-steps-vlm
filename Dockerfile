# The camera-box image: ONNX Runtime and the edge service, no PyTorch. Build after `lsv evaluate` has exported the
# models to runs/dev-full/onnx.
FROM python:3.13-slim
WORKDIR /app
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt
COPY lsv/__init__.py lsv/
COPY lsv/product/__init__.py lsv/product/edge.py lsv/product/checks.py lsv/product/tracker.py lsv/product/service.py lsv/product/
COPY runs/dev-full/onnx models/
RUN useradd -m box
USER box
EXPOSE 8080
CMD ["python", "-c", "import uvicorn; from pathlib import Path; from lsv.product.service import create_app; uvicorn.run(create_app(Path('models')), host='0.0.0.0', port=8080)"]
