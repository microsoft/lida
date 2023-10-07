# Use an official Python runtime as a parent image 
FROM python:3.10-slim

# Set environment variables 
ENV PTYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

# Add open api key to the env
ENV OPENAI_API_KEY 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

# Set the working directory in the container 
WORKDIR /app

# Copy the requirements file into the container 
COPY requirements.txt /app/

# Install any needed packages specified in the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container 
COPY . /app/ 

# Expose the port that the application will listen on 
EXPOSE 8080

# Start the Web UI
CMD ["lida", "ui", "--host", "0.0.0.0", "--port", "8080", "--docs"]