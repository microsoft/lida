# Use an official Python runtime as a parent image 
FROM python:3.10-slim

# Set environment variables 
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

# Set the working directory in the container 
WORKDIR /app

# Install requirements
RUN pip install --no-cache-dir lida

# Expose the port that the application will listen on 
EXPOSE 8080

# Start the Web UI
CMD ["lida", "ui", "--host", "0.0.0.0", "--port", "8080", "--docs"]