Here’s the updated version of the instructions, including steps for importing and exporting data, with a note about the migration folder:

---

## **Updated Project Setup and Running Instructions**

### **Prerequisites**
- Python 3.8 or higher installed on your system.
- `pip` (Python package manager) installed.
- **Ollama** installed for running LLM models locally.
- **Appwrite** installed and configured.

---

### **Installation Steps**

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:vinayakyadav2709/00101010-Kisanverse.git
   cd 00101010-Kisanverse
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**:
   - Follow the instructions to install Ollama from the [official website](https://ollama.com/).
   - After installation, download the required model:
     ```bash
     ollama pull gemma3:4b-it-q8_0
     ```

5. **Install Appwrite**:
   - Follow the [Appwrite installation guide](https://appwrite.io/docs/installation) to set up Appwrite locally or on a server.
   - Create a project in Appwrite and note the **Project ID** and **API Key**.

6. **Set Environment Variables**:
   - Update the .env file in the root directory with the following variables:
     ```properties
     # Ollama Configuration
     OLLAMA_BASE_URL=http://localhost:11434
     OLLAMA_MODEL=gemma3:4b-it-q8_0

     # Appwrite Configuration
     APPWRITE_API_KEY=your_appwrite_api_key
     APPWRITE_PROJECT_ID=your_appwrite_project_id
     APPWRITE_ENDPOINT=http://localhost/v1  # Update this if Appwrite is hosted elsewhere
     ```

---

### **Running the Application**

1. **Navigate to the app Directory**:
   ```bash
   cd app
   ```

2. **Start the Application Using `uvicorn`**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Open Your Browser and Navigate to**:
   ```
   http://127.0.0.1:8000
   ```

---

### **Data Migration (Export and Import)**

#### **Export Data**
If you want to export data from your current Appwrite instance (e.g., local instance), follow these steps:
1. Ensure your .env file is configured to point to the source Appwrite instance.
2. Run the export script:
   ```bash
   python export_data.py
   ```
3. This will generate:
   - `collections_data.json`: Contains all documents from the collections.
   - `buckets_data.json`: Contains metadata of all files in the buckets.
   - Files downloaded locally with names like `bucket_name_file_name`.

#### **Import Data**
If you want to import data into a new Appwrite instance (e.g., cloud instance), follow these steps:
1. Ensure your .env file is configured to point to the target Appwrite instance.
2. Copy the exported files (`collections_data.json`, `buckets_data.json`, and downloaded files) to the `migration` folder in the project root.
3. Run the import script:
   ```bash
   python import_data.py
   ```
4. This will recreate the collections, documents, and files in the target Appwrite instance.

---

### **Key Constants to Update in `config.py`**

1. **Ollama Configuration**:
   - Ensure the following constants are set in `config.py`:
     ```python
     OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b-it-q8_0")
     OLLAMA_DEFAULT_URL = "http://localhost:11434"
     OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL)
     ```

2. **Appwrite Configuration**:
   - Ensure the Appwrite environment variables are set in .env (endpoint depends on where Appwrite is hosted):
     ```properties
     APPWRITE_API_KEY=your_appwrite_api_key
     APPWRITE_PROJECT_ID=your_appwrite_project_id
     APPWRITE_ENDPOINT=http://localhost/v1
     ```

---

### **Migration Folder**
- The `migration` folder in the project root is used to store exported data (`collections_data.json`, `buckets_data.json`, and files).
- Ensure this folder is accessible when running the import script.

---

### **Additional Notes**
- Replace `your_appwrite_api_key` and `your_appwrite_project_id` with the actual values from your Appwrite project.
- Ensure the `OLLAMA_BASE_URL` is reachable (default: `http://localhost:11434`).
- Use `--reload` for development purposes to enable auto-reloading on code changes.
- Verify the data after migration to ensure all collections, documents, and files are transferred correctly.
- app/models/weights folder is missing, which is needed to run the models. you can request it through github