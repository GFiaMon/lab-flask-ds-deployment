# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p insurance_frontend

# Expose the port Flask runs on
EXPOSE 5001

# Define environment variable
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python", "app_insurance.py"]