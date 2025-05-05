# Kisanverse Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kisanverse aims to provide an AI-powered suite of tools for farmers, including a voice assistant for interaction, a marketplace backend, crop prediction capabilities, and administrative/mobile interfaces.

## Features

*   **AI Voice Assistant (`AICallAssistant`):** A Flask-based application providing a conversational interface via voice (integrates with Twilio, STT, LLM).
*   **Backend API (`backend`):** A FastAPI/Uvicorn application serving as the core API for marketplace listings, contracts, subsidies, and potentially other data interactions.
*   **Admin Panel (`admin`):** A Next.js web application for managing platform data and users.
*   **Mobile Application (`mobile-app`):** An Expo (React Native) application for farmer/user interaction.
*   **Crop Prediction (`Crop_prediction`):** Contains models and scripts related to crop yield or suitability predictions (likely used by the backend).
*   **Marketplace Functionality:** Includes features for listing crops, placing bids, managing contracts, and viewing subsidies (powered by the `backend` and potentially visualized in `admin` and `mobile-app`).

## Directory Structure
Use code with caution.
Markdown
.
├── AICallAssistant/ # Flask Voice Assistant code
├── admin/ # Next.js Admin Panel code
├── backend/ # FastAPI/Uvicorn Backend API code
├── Crop_prediction/ # Crop prediction models/scripts
├── mobile-app/ # Expo Mobile Application code
├── all_data.json # Static data for marketplace (used by AICallAssistant backend_functions)
├── recommendation_output.json # Static data for crop recommendations (used by AICallAssistant backend_functions)
├── ss.json # Static soil/weather data (used by AICallAssistant backend_functions)
├── weather_list.json # Static daily weather forecast data (used by AICallAssistant backend_functions)
├── PredictionReport.md # Markdown report related to predictions
├── README.md # This file
├── LICENSE # Project License (MIT)
├── CONTRIBUTING.md # Contribution Guidelines
├── docker-compose.yml # Docker Compose configuration
├── .env.example # Example environment variables
└── .gitignore # Git ignore rules
## Prerequisites

Before you begin, ensure you have the following installed:

*   **Git:** For cloning the repository.
*   **Docker & Docker Compose:** For running the application using containers (Recommended method).
*   **Python:** (>= 3.10 recommended) For running services individually or for dependency installation if not using Docker for everything.
*   **Node.js:** (>= 18.x recommended) Required for the `admin` (Next.js) and `mobile-app` (Expo) development.
*   **npm or yarn:** Node package manager.
*   **Ollama:** (Optional, if using `LLM_PROVIDER=ollama`) Needs to be installed and running locally if not using the Docker Compose `ollama` service. [https://ollama.com/](https://ollama.com/)
*   **Expo CLI:** For running the mobile application locally. (`npm install -g expo-cli` or `yarn global add expo-cli`)
*   **System Dependencies for Voice Assistant:**
    *   `ffmpeg`: Required by `openai-whisper`. (e.g., `sudo apt update && sudo apt install ffmpeg`)
    *   `mpg123`: Required by the simulator (`twilio_simulator.py`) for TTS playback and MP3->WAV conversion. (e.g., `sudo apt update && sudo apt install mpg123`)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:vinayakyadav2709/00101010-Kisanverse.git
    cd 00101010-Kisanverse
    ```

2.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   **Edit the `.env` file** and fill in your actual credentials and settings for:
        *   `LLM_PROVIDER` (e.g., `e2e` or `ollama`)
        *   E2E LLM/Whisper Credentials (if applicable)
        *   Twilio Credentials (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`)
        *   Any backend-specific variables (like `DATABASE_URL`, if needed).
        *   Refer to the comments in `.env.example` for details.
    *   **Important:** Do NOT commit your actual `.env` file to version control.

3.  **(Optional) Build Docker Images:** While `docker compose up` will build images if they don't exist, you can pre-build them:
    ```bash
    docker compose build
    ```

## Running the Application (Recommended: Docker Compose)

This method runs the Voice Assistant, Backend API, Admin Panel, and optionally Ollama in separate containers.

1.  **Ensure Docker Desktop or Docker Engine with Compose is running.**
2.  **Ensure Ollama models are pulled (if using Ollama):** If `LLM_PROVIDER=ollama` in your `.env`, make sure the model specified (`OLLAMA_MODEL`) is pulled locally *before* starting compose if you aren't relying solely on the container's volume persistence yet:
    ```bash
    ollama pull <your_model_name>
    # e.g., ollama pull gemma3:4b-it-q8_0
    ```
3.  **Start the services:**
    ```bash
    docker compose up -d
    ```
    (Use `docker-compose up -d` if you have the older standalone version). The `-d` flag runs containers in detached mode.

4.  **Access Services:**
    *   **Admin Panel:** Open your browser to [http://localhost:3000](http://localhost:3000)
    *   **Backend API:** Accessible at `http://localhost:8000`. Check its documentation (often at `http://localhost:8000/docs` for FastAPI) for available endpoints.
    *   **Voice Assistant API:** Accessible at `http://localhost:5000`. Use this URL in the Twilio simulator or your Twilio phone number configuration (e.g., webhook for `/voice`).
    *   **Ollama API (if running in compose):** Accessible at `http://localhost:11434`.

5.  **View Logs:**
    ```bash
    docker compose logs -f # View logs for all services
    docker compose logs -f voice-assistant # View logs for a specific service
    ```

6.  **Stop the services:**
    ```bash
    docker compose down
    ```
    Use `docker compose down -v` to also remove associated volumes (like the Ollama model cache).

## Running Services Individually (Alternative)

If you prefer not to use Docker, you can run each service separately. Ensure all prerequisites are installed on your host machine. You'll need separate terminal windows for each service.

**1. Backend API (`backend` directory):**

```bash
cd backend
# Create virtual env (optional but recommended)
# python -m venv venv && source venv/bin/activate
pip install -r requirements.txt # Make sure this file exists
# Set any required backend environment variables (e.g., DATABASE_URL)
export DATABASE_URL="your_db_connection_string" # Example
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Use code with caution.
2. AI Voice Assistant (AICallAssistant directory):
cd AICallAssistant
# Create virtual env (optional but recommended)
# python -m venv venv && source venv/bin/activate
pip install -r requirements.txt # Make sure this file exists
# Ensure Ollama is running locally if LLM_PROVIDER=ollama
# Ensure E2E credentials are set if LLM_PROVIDER=e2e
# Ensure Twilio credentials are set
# Ensure STT Provider selected is set up (Whisper model downloaded or E2E creds set)
uv run app.py # Or: flask run --host=0.0.0.0 --port=5000
Use code with caution.
Bash
3. Admin Panel (admin directory):
cd admin
npm install # or yarn install
# Set the backend API URL environment variable for the frontend
export NEXT_PUBLIC_API_URL="http://localhost:8000" # Point to backend running locally
npm run dev # or yarn dev
Use code with caution.
Bash
Access at http://localhost:3000.
Running the Mobile App (mobile-app directory)
The Expo mobile app is typically run using the Expo CLI for development.
Navigate to the directory:
cd mobile-app
Use code with caution.
Bash
Install dependencies:
npm install
# or
yarn install
Use code with caution.
Bash
Start the development server:
npx expo start
Use code with caution.
Bash
Follow the instructions in the terminal:
Scan the QR code with the Expo Go app on your physical device.
Or press a for Android emulator, i for iOS simulator (if configured).
Ensure the mobile app is configured to point to your backend API (likely http://<your-local-ip>:8000 if running the backend locally, or the appropriate deployed URL).
Environment Variables
Key environment variables are configured in the .env file. Refer to .env.example for a full list and descriptions. Main variables include:
LLM_PROVIDER: Selects the Language Model service (ollama or e2e).
STT_PROVIDER: Selects the Speech-to-Text service (whisper or e2e_whisper).
*_HOST, *_MODEL, *_API_KEY, etc.: Credentials and endpoints for selected services.
TWILIO_*: Twilio account credentials.
DATABASE_URL: (Example) Connection string for the backend database.
NEXT_PUBLIC_API_URL: Used by the admin frontend to locate the backend API (set automatically in docker-compose).
License
This project is licensed under the MIT License - see the LICENSE file for details.