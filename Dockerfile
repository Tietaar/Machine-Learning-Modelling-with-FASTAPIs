# Use Python 3.10.8 base image
FROM python:3.10.8


# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn joblib pandas xgboost imbalanced-learn

# Expose the port that FastAPI runs on
EXPOSE 8080

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
