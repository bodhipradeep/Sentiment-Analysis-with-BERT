FROM python:3.11

WORKDIR /app

COPY . ./

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

