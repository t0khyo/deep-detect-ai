# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt (from the app directory)
RUN pip install -r app/requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable for Flask to run
ENV FLASK_APP=app.main

# Run the Flask app using gunicorn (recommended for production)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app.main:app"]
