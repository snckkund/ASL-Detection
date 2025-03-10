FROM python:3.9

WORKDIR /app

COPY backend/ ./backend
COPY frontend/ ./frontend

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]