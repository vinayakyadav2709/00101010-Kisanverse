
# Project Setup and Running Instructions

## Prerequisites
- Python 3.8 or higher installed on your system.
- `pip` (Python package manager) installed.

## Installation Steps
1. Clone the repository:
   ```bash
   git clone git@github.com:vinayakyadav2709/00101010-Kisanverse.git
   cd 00101010-Kisanverse
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Running the Application
1. Navigate to the `app` directory:
   ```bash
   cd app
   ```

2. Start the application using `uvicorn`:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

## Additional Notes
- Use `--reload` for development purposes to enable auto-reloading on code changes.
- Ensure all environment variables (if any) are properly configured before running the application.

