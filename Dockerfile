FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# The app is standard-library only; embedder.py (PyMuPDF) runs outside the image.
COPY rag.py lambda_function.py local_server.py faq_meta.json ./
COPY web ./web

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

CMD ["python", "local_server.py"]
