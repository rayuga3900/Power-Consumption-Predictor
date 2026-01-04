# loading Linux os + python preinstalled
FROM python:3.9-slim 

# creating folder in container to hold the application
WORKDIR /app

# copying requirements.txt from local to container
COPY requirements.txt . 

# Install Python dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt


# Copy the entire project code into the container
COPY /app .

# Inform Docker that the container will use port 8501
EXPOSE 8501

# Command to run Streamlit when the container starts
# --server.port=8501 → Streamlit listens on port 8501 inside the container
# --server.address=0.0.0.0 → Streamlit listens on all network interfaces inside the container
# This allows you to access the app from your host machine when using port mapping (-p)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

 