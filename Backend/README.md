# Airbnb Room Type Predictor - Backend

A robust machine learning backend service built with **FastAPI** and **scikit-learn** that predicts Airbnb room types based on listing features (such as location, price, minimum nights, and reviews).

---

## 📂 Project Structure

```text
Backend/
│
├── Datasets/
│   ├── Cleaned/
│   │   └── Cleaned_data.csv       # Preprocessed and cleaned dataset
│   └── RAW/
│       └── AB_NYC_2019.csv        # Original raw Airbnb dataset
│
├── Notebooks/
│   ├── Data_cleaning.ipynb        # Jupyter notebook for initial data cleaning
│   ├── EDA.ipynb                  # Exploratory Data Analysis notebook
│   └── Preprocessing and...       # Feature engineering and model training notebook
│
├── src/
│   └── data_cleaning.py           # Reusable data cleaning pipeline functions
│
├── venv/                          # Python virtual environment
├── main.py                        # FastAPI application entry point
├── Model_Pipeline.pkl             # Serialized trained machine learning model pipeline
└── requirements.txt               # Project dependencies
```

---

## 🚀 Features

* **FastAPI Web Service:** Lightning-fast API endpoints with automatic interactive documentation (Swagger UI).
* **ML Model Integration:** Uses a pre-trained `joblib` pipeline (`Model_Pipeline.pkl`) to run real-time inferences.
* **Pydantic Validation:** Strict payload validation to ensure input types, ranges, and data integrity before processing.
* **CORS Middleware Enabled:** Fully configured to accept requests from any frontend origin (e.g., React.js).

---

## 🛠️ Installation & Setup

### 1. Clone the repository & navigate to the backend folder
```bash
cd Backend
```

### 2. Create and activate a virtual environment
* **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Application

Start the FastAPI development server using `uvicorn`:

```bash
uvicorn main:app --reload
```

* The server will run locally at: `http://127.0.0.1:8000`
* **Interactive API Docs (Swagger UI):** Open your browser and go to `http://127.0.0.1:8000/docs`

---

## 🔌 API Endpoints

### 1. Root Check
* **URL:** `/`
* **Method:** `GET`
* **Response:** `{"Greet": "Hello users!"}`

### 2. Predict Room Type
* **URL:** `/predict`
* **Method:** `POST`
* **Request Body Example (JSON):**
  ```json
  {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "price": 150.0,
    "minimum_nights": 2,
    "number_of_reviews": 25,
    "reviews_per_month": 1.5,
    "calculated_host_listings_count": 1,
    "availability_365": 300,
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Midtown"
  }
  ```
* **Response Example:**
  ```json
  {
    "Predicted_room_type": "Entire home/apt",
    "Probability": [0.85, 0.10, 0.05]
  }
  ```