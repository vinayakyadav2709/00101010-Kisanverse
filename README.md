
# Kisanverse Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kisanverse aims to provide an AI-powered suite of tools for farmers, including a voice assistant for interaction, a marketplace backend, crop prediction capabilities, and administrative/mobile interfaces.


**➡️ View the [AI System Technical Report (PredictionReport.md)](./PredictionReport.md)**

## Features

*   **AI Voice Assistant (`AICallAssistant`):** A Flask-based application providing a conversational interface via voice (integrates with Twilio, STT, LLM). Handles core voice interaction logic and uses backend functions for data operations.
*   **Backend API (`backend`):** A FastAPI/Uvicorn application serving as the core API for marketplace listings, contracts, subsidies, user data, and potentially other data interactions. Intended to be consumed by the `admin` panel and `mobile-app`.
*   **Admin Panel (`admin`):** A Next.js web application for managing platform data and users via the `backend` API.
*   **Mobile Application (`mobile-app`):** An Expo (React Native) application for farmer/user interaction, communicating with the `backend` API.
*   **Crop Prediction (`Crop_prediction`):** Contains models and scripts related to crop yield or suitability predictions (likely integrated into or called by the `backend` API).
*   **Marketplace Functionality:** Features for listing crops, placing bids, managing contracts, and viewing subsidies are primarily handled by the `backend` API, with interfaces provided by the `admin`, `mobile-app`, and potentially queried via the `AICallAssistant`.

## Directory Structure

```
.
├── AICallAssistant/      # Flask Voice Assistant code + its backend_functions.py
│   ├── all_data.json         # Static data (used by AICallAssistant)
│   ├── recommendation_output.json # Static data (used by AICallAssistant)
│   ├── ss.json               # Static data (used by AICallAssistant)
│   ├── weather_list.json     # Static data (used by AICallAssistant)
│   ├── Dockerfile            # Dockerfile for this service
│   └── ... (other python files: app.py, config.py, etc.)
├── admin/                # Next.js Admin Panel code
│   ├── Dockerfile            # Dockerfile for this service
│   └── ... (next.js files)
├── backend/              # FastAPI/Uvicorn Backend API code
│   ├── Dockerfile            # Dockerfile for this service
│   └── ... (fastapi files, main.py)
├── Crop_prediction/      # Crop prediction models/scripts (if separate)
├── mobile-app/           # Expo Mobile Application code
├── PredictionReport.md   # Markdown report related to predictions
├── README.md             # This file
├── LICENSE               # Project License (MIT)
├── CONTRIBUTING.md       # Contribution Guidelines
├── docker-compose.yml    # Docker Compose configuration
├── .env.example          # Example environment variables
└── .gitignore            # Git ignore rules
```

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Git:** For cloning the repository.
*   **Docker & Docker Compose:** For running the application using containers (Recommended method).
*   **Python:** (>= 3.10 recommended) For running services individually or for initial setup if not using Docker for everything.
*   **Node.js:** (>= 18.x recommended) Required for the `admin` (Next.js) and `mobile-app` (Expo) development.
*   **npm or yarn:** Node package manager.
*   **Ollama:** (Optional, if using `LLM_PROVIDER=ollama`) Needs to be installed and running locally if not using the Docker Compose `ollama` service. [https://ollama.com/](https://ollama.com/)
*   **Expo CLI:** For running the mobile application locally. (`npm install -g expo-cli` or `yarn global add expo-cli`)
*   **System Dependencies for Voice Assistant Container/Host:**
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
    *   **Edit the `.env` file** and fill in your actual credentials and settings. Refer to the comments in `.env.example` for required variables (LLM Provider, STT Provider, Twilio keys, E2E keys if used, etc.).
    *   **Important:** Do NOT commit your actual `.env` file to version control. The `.gitignore` file should prevent this.

3.  **(Optional) Build Docker Images:** While `docker compose up` builds automatically, you can pre-build:
    ```bash
    docker compose build
    ```

## Running the Application (Recommended: Docker Compose)

This method runs the Voice Assistant, Backend API, Admin Panel, and optionally Ollama in containers.

1.  **Start Docker:** Ensure Docker Desktop or Docker Engine with Compose is running.
2.  **Pull Ollama Model (if using Ollama in `.env`):** Make sure the `OLLAMA_MODEL` specified in `.env` is pulled locally if you want to use it immediately (Compose volume will persist it later):
    ```bash
    ollama pull <your_model_name_from_env>
    # e.g., ollama pull gemma3:4b-it-q8_0
    ```
3.  **Start Services:** From the project root directory:
    ```bash
    docker compose up -d
    ```
    (`-d` runs in detached mode).

4.  **Access Services:**
    *   **Admin Panel:** [http://localhost:3000](http://localhost:3000)
    *   **Backend API:** `http://localhost:8000` (Check `/docs` for Swagger UI if using FastAPI)
    *   **Voice Assistant API:** `http://localhost:5000` (Use for Twilio webhooks/simulator)
    *   **Ollama API (if running in compose):** `http://localhost:11434`

5.  **View Logs:**
    ```bash
    docker compose logs -f                # Tail logs for all services
    docker compose logs -f voice-assistant # Tail logs for a specific service
    ```

6.  **Stop Services:**
    ```bash
    docker compose down                  # Stop and remove containers, networks
    # docker compose down -v               # Stop and remove containers, networks, AND volumes (e.g., Ollama model cache)
    ```

## Running Services Individually (Alternative)

Run each service in a separate terminal. Ensure host prerequisites are met.

**1. Backend API (`backend` directory):**

```bash
cd backend
# Recommended: python -m venv venv && source venv/bin/activate
pip install -r requirements.txt # Ensure file exists
# export DATABASE_URL="..." # Set necessary ENV VARS
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**2. AI Voice Assistant (`AICallAssistant` directory):**

```bash
cd AICallAssistant
# Recommended: python -m venv venv && source venv/bin/activate
pip install -r requirements.txt # Ensure file exists
# Ensure .env file is present in project root or required VARS are exported
# Ensure Ollama server running if LLM_PROVIDER=ollama
# Ensure required system packages (ffmpeg, mpg123) are installed
uv run app.py
# Or: flask run --host=0.0.0.0 --port=5000
```
*Note: Ensure `.env` is in the root (`00101010-Kisanverse`), and `AICallAssistant/config.py` loads it correctly (using `load_dotenv()` without path usually works if run from root, adjust if needed).*

**3. Admin Panel (`admin` directory):**

```bash
cd admin
npm install # or yarn install
export NEXT_PUBLIC_API_URL="http://localhost:8000" # Point to locally running backend
npm run dev # or yarn dev
```
Access at [http://localhost:3000](http://localhost:3000).

**4. Mobile App (`mobile-app` directory):**

```bash
cd mobile-app
npm install # or yarn install
npx expo start
```
Follow terminal instructions to connect via Expo Go or emulators. Ensure the app's API configuration points to your locally running backend (`http://<YOUR_LOCAL_IP>:8000`).

## Environment Variables

Key environment variables are configured in the project's root `.env` file (copied from `.env.example`). These control service credentials, API endpoints, and feature selections.

*   **`LLM_PROVIDER`**: `ollama` or `e2e`.
*   **`STT_PROVIDER`**: `whisper` or `e2e_whisper`.
*   **`OLLAMA_*` / `E2E_*` / `E2E_TIR_*`**: Service-specific credentials and model names.
*   **`TWILIO_*`**: Twilio account credentials.
*   **`DATABASE_URL`**: (Example) Connection string for the backend database.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

