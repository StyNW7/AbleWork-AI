# AbleWork - AI Repository

This repository contains the AI-related code for the AbleWork application.

## Prerequisites

Before setting up the project, make sure you have the following tools and software installed:

- **Python 3.x** (recommended: Python 3.8 or higher)
- **pip** (Python package installer)

## Installation

Follow these steps to set up the AI components locally:

### 1. **Set Up Python Virtual Environment**:

To ensure the project runs with the correct dependencies, it is recommended to use a virtual environment. Follow these steps:

- Create a virtual environment:
  ```bash
  python -m venv venv
  ```

- Activate the virtual environment:

  On Windows:
  ```bash
  venv\Scripts\activate
  ```

  On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 2. **Install Python Dependencies**:

Once the virtual environment is activated, you can install the required Python dependencies.

- Using `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

- Alternatively, you can install the dependencies manually:
  ```bash
  pip install flask
  pip install flask-cors
  pip install pymupdf
  pip install scikit-learn
  pip install huggingface_hub
  pip install python-dotenv
  ```

### 3. **Set Up Environment Variables**:

Create a `.env` file in the root directory to store sensitive information like your Hugging Face API key and other configuration details.

Example `.env` file:
```ini
HF_API_KEY=your_huggingface_api_key_here
```

Replace the placeholder with your Hugging Face API key.

### 4. **Start the AI Server**:

Once the dependencies are installed and the `.env` file is set up, you can start the AI server by running the following:

```bash
python main.py
```

By default, the server will be running on `http://localhost:5000`.

### 5. **Deactivate Virtual Environment**:

After finishing your work, you can deactivate the virtual environment by running:

```bash
deactivate
```