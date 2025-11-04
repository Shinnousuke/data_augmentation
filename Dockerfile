# -------------------------------
# 1️⃣ Base Image
# -------------------------------
FROM python:3.10-slim

# -------------------------------
# 2️⃣ Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 3️⃣ Copy files to container
# -------------------------------
COPY . /app

# -------------------------------
# 4️⃣ Install system dependencies (for PIL / numpy / image processing)
# -------------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 5️⃣ Install Python dependencies
# -------------------------------
RUN pip install --no-cache-dir streamlit pillow numpy

# -------------------------------
# 6️⃣ Expose the port for Cloud Run
# -------------------------------
EXPOSE 8080

# -------------------------------
# 7️⃣ Set environment variables for Streamlit
# -------------------------------
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# -------------------------------
# 8️⃣ Run the app
# -------------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
