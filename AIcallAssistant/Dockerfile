# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any - e.g., ffmpeg, mpg123 needed by handlers/simulator)
# Adjust based on actual needs of your Python code and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    mpg123 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application (using uv as per previous examples)
# Use 0.0.0.0 to make it accessible outside the container
CMD ["uv", "run", "app.py"]

# OR if using Flask directly:
# CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]