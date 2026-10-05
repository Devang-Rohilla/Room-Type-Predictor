# Airbnb Room Type Predictor - Full Stack Project

A complete end-to-end machine learning web application featuring a high-performance **FastAPI** backend and a modern **React.js & Tailwind CSS** glassmorphic frontend.

---

## 📂 Complete Project Structure

```text
Airbnb Room Type Predictor/
│
├── Backend/
│   ├── Datasets/
│   │   ├── Cleaned/
│   │   │   └── Cleaned_data.csv       # Preprocessed dataset
│   │   └── RAW/
│   │       └── AB_NYC_2019.csv        # Original raw dataset
│   ├── Notebooks/
│   │   ├── Data_cleaning.ipynb        # Data cleaning workflow
│   │   ├── EDA.ipynb                  # Exploratory data analysis
│   │   └── Preprocessing and pipeline.ipynb # Model training & export
│   ├── src/
│   │   └── data_cleaning.py           # Reusable data cleaning pipeline
│   ├── venv/                          # Python virtual environment
│   ├── main.py                        # FastAPI application entry point
│   ├── Model_Pipeline.pkl             # Trained scikit-learn model pipeline
│   └── requirements.txt               # Python dependencies
│
└── Frontend/
    ├── node_modules/
    ├── src/
    │   ├── App.jsx                    # Main React component & UI logic
    │   ├── index.css                  # Tailwind CSS imports & styles
    │   └── main.jsx                   # React application entry point
    ├── index.html                     # HTML template
    ├── package.json                   # Node dependencies & scripts
    ├── postcss.config.js              # PostCSS configuration
    └── tailwind.config.js             # Tailwind CSS configuration
```

---

## 🛠️ Getting Started & Setup

To run the full stack application locally, you will need two separate terminal windows (one for the Backend server and one for the Frontend client).

### Part 1: Backend Setup (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd Backend
   ```
2. Create and activate your virtual environment:
   * **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   * **macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   * *The API will be running at:* `http://localhost:8000`
   * *Interactive API Docs:* `http://localhost:8000/docs`

---

### Part 2: Frontend Setup (React & Tailwind)

1. Open a new terminal window and navigate to the frontend directory:
   ```bash
   cd Frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   * *The web app will run locally at:* `http://localhost:5173` (or the port specified by Vite).

---

## 🔌 API Integration Details

* **Endpoint:** `http://localhost:8000/predict`
* **Method:** `POST`
* **Payload Format (JSON):**
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