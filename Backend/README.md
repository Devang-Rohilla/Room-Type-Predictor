# Airbnb Room Type Predictor - Machine Learning & Backend API

An enterprise-grade, high-performance machine learning backend service engineered to classify Airbnb listings into their respective room categories (`Entire home/apt`, `Private room`, or `Shared room`). Powered by **FastAPI** and an optimized **Scikit-Learn** ensemble pipeline, this service processes spatial, temporal, pricing, and host behavioral signals to deliver real-time multi-class predictions and calibrated posterior probabilities with sub-35ms inference latency.

---

## 1. Project Title & Overview

### Problem Statement
In peer-to-peer property rental marketplaces, listing categorizations frequently suffer from host mislabeling, ambiguity during self-onboarding, or deliberate classification bypass to circumvent local municipal ordinances (e.g., New York City's Local Law 18 short-term rental restrictions). Accurately categorizing accommodations is critical for search indexing, dynamic pricing models, tax assessments, and regulatory compliance auditing.

### Solution Architecture
This repository provides a dedicated machine learning backend service built around an end-to-end serialized classification pipeline (`Model_Pipeline.pkl`). The service ingests listing metadata—such as geographical coordinates, nightly tariff, minimum stay duration, historical review velocity, host portfolio size, and annual calendar availability—and computes the optimal room category classification alongside class-specific confidence distributions. The system is served via an asynchronous ASGI architecture using FastAPI and Uvicorn, strictly validating incoming payloads via Pydantic schema contracts.

---

## 2. Features

- **Automated Multi-Class Inference**: Accurately classifies listings across three discrete accommodation tiers: `Entire home/apt`, `Private room`, and `Shared room`.
- **Calibrated Posterior Probabilities**: Generates confidence vectors across all supported classes, enabling downstream systems to implement risk thresholds and fallback rules.
- **Unified Pipeline Architecture**: Couples preprocessing, feature imputation, scaling, one-hot encoding, and ensemble classification into an immutable serialized artifact (`joblib`), eliminating training-serving skew.
- **Strict Runtime Validation**: Implements Pydantic v2 schemas enforcing geographical boundaries (latitude, longitude), positive price constraints, calendar limits (1–365 days), and required categorical fields.
- **Cost-Sensitive Class Balancing**: Incorporates balanced sample weighting to mitigate severe minority class imbalance (Shared Room represents only 2.37% of the dataset) without synthetic sample distortion.
- **Asynchronous, Low-Latency REST API**: Built on FastAPI and Starlette, delivering benchmarked inference latencies under 35 milliseconds.
- **Automated API Documentation**: Native OpenAPI 3.1.0 integration providing automated interactive Swagger UI (`/docs`) and ReDoc (`/redoc`) documentation.
- **CORS-Compliant Middleware**: Configured with permissive Cross-Origin Resource Sharing policies for integration into distributed microservice topologies.

---

## 3. Data Processing & EDA

The underlying models were developed using the **New York City Airbnb Open Data (2019)** dataset, comprising 48,895 listing records and 16 heterogeneous attributes.

### Exploratory Data Analysis (EDA) Insights
- **Target Distribution & Severe Class Imbalance**:
  - `Entire home/apt`: 25,409 listings (51.97%)
  - `Private room`: 22,326 listings (45.66%)
  - `Shared room`: 1,160 listings (2.37%)
  The extreme scarcity of `Shared room` listings required specialized cost-sensitive weighting to prevent majority-class collapse.
- **Spatial Clustering**: Strong geographic concentration was observed across borough groups (`Manhattan` and `Brooklyn` comprise >85% of listings). Spatial mapping confirmed that `Entire home/apt` inventory is heavily concentrated in Midtown and Lower Manhattan, whereas `Private room` listings are widely dispersed across Upper Manhattan, Brooklyn, and Queens.
- **Price Distribution Dynamics**: Nightly prices exhibited extreme positive skewness (range: \$0 to \$10,000, with a median of \$106 and mean of \$152.72). `Entire home/apt` demonstrated significantly higher median valuations compared to private and shared alternatives.
- **Feature Interdependence**: A moderate positive correlation (\(r = 0.55\)) exists between `number_of_reviews` and `reviews_per_month`. Weak linear correlations between individual numeric attributes and the target confirmed that room type differentiation depends on non-linear feature interactions.

### Data Cleaning Workflow (`data_cleaning.py`)
1. **Identifier & Metadata Pruning**: Dropped high-cardinality, non-generalizable metadata columns:
   - `id`, `name`, `host_id`, `host_name`, and timestamp `last_review`.
2. **Missing Value Resolution**:
   - `reviews_per_month` contained 10,052 null values, perfectly corresponding to listings with `number_of_reviews == 0`. These were imputed with `0.0` rather than synthetic statistics to preserve real-world domain meaning.
3. **Outlier Capping (Winsorization)**:
   - Extreme long-tail outliers were clamped at the 99th percentile (`quantile(0.99)`) for `price` and `minimum_nights` to protect tree split heuristics and variance estimators without data loss.

### Feature Preprocessing & Transformation Pipeline
The dataset was split using a stratified 67/33 ratio (32,759 training samples, 16,136 testing samples) to preserve class distributions. Transformations were wrapped in a `ColumnTransformer`:
- **Numerical Pipeline** (`latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`):
  1. `SimpleImputer(strategy='median')`: Robust against residual anomalies.
  2. `StandardScaler()`: Zero-mean, unit-variance standardization.
- **Categorical Pipeline** (`neighbourhood_group`, `neighbourhood`):
  1. `SimpleImputer(strategy='most_frequent')`: Handles missing category labels.
  2. `OneHotEncoder(handle_unknown='ignore')`: Converts nominal categories into sparse binary vectors while handling previously unseen categories in inference payloads.

---

## 4. Machine Learning Algorithms

Four candidate algorithms were trained and evaluated during systematic 3-fold cross-validation on the training set, comparing balanced versus unweighted heuristics.

```
                           Raw Listing Metadata
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │     Pydantic Validation      │
                     │  (Boundaries, Types, Nulls)  │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   scikit-learn Preprocessor  │
                     │  ├── Numeric: Median + Scale │
                     │  └── Categoric: Impute + OHE │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │    RandomForestClassifier    │
                     │  (150 Trees, Balanced Class  │
                     │   Weights, min_samples=5)    │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                    Predicted Class & Probability Vector
```

1. **Logistic Regression (Linear Baseline)**:
   - Configuration: `class_weight='balanced'`, `random_state=42`.
   - Role: Established a baseline for linear separability. Yielded sub-optimal macro-F1 (0.522) due to inability to model non-linear spatial-price boundaries.
2. **Decision Tree Classifier**:
   - Configuration: `class_weight='balanced'`, `random_state=42`.
   - Role: Non-linear single-tree baseline. Captured localized feature partitions but showed high variance and prone to overfitting (Macro-F1: 0.648).
3. **Gradient Boosting Classifier (Ensemble - Boosting)**:
   - Configuration: `n_estimators=100`, `learning_rate=0.1`, `random_state=42`.
   - Role: Sequential gradient-boosted trees. Achieved high raw accuracy (85.0%) but suffered on minority class recall (Macro-F1: 0.705) due to lack of native sample weighting support in scikit-learn's implementation.
4. **Random Forest Classifier (Selected Production Model)**:
   - Configuration: `class_weight='balanced'`, `n_estimators=150`, `min_samples_split=5`, `max_depth=None`, `random_state=42`.
   - Role: Bagged ensemble combining multiple de-correlated randomized decision trees. Optimized via `RandomizedSearchCV` over parameter grids (`n_estimators`, `max_depth`, `min_samples_split`) targeting `f1_macro`. Delivers superior generalization, robust noise suppression, and balanced multi-class performance.

---

## 5. Scores & Evaluation Metrics

Given the significant representation skew (Shared rooms accounting for under 2.4% of total inventory), **Macro-Averaged F1-Score** was prioritized alongside **Overall Accuracy** to prevent metric inflation.

### Primary Validation & Test Metrics
- **Holdout Test Accuracy**: `85.47%` (0.85467)
- **Holdout Macro-F1 Score**: `74.30%` (0.74300)
- **Holdout Weighted-F1 Score**: `85.28%` (0.8528)
- **Best 3-Fold Cross-Validation Macro-F1**: `73.29%` (0.73286)
- **Mean API Inference Latency**: `33.12 ms` (Min: `28.52 ms` per single-row prediction)

### Algorithm Benchmark Comparison (3-Fold Stratified Cross-Validation)

| Algorithm | Class Weight Strategy | CV Accuracy | CV Macro F1 |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Balanced | 65.9% | 0.522 |
| **Decision Tree** | Balanced | 78.2% | 0.648 |
| **Gradient Boosting** | Default (Unweighted) | 85.0% | 0.705 |
| **Random Forest (Tuned)** | **Balanced** | **85.5%** | **0.733** |

### Detailed Classification Report (Test Set, N = 16,136)

| Target Class | Precision | Recall | F1-Score | Support (N) |
| :--- | :--- | :--- | :--- | :--- |
| **Entire home/apt** | 0.8760 | 0.8732 | 0.8746 | 8,385 |
| **Private room** | 0.8340 | 0.8579 | 0.8458 | 7,368 |
| **Shared room** | 0.7437 | 0.3864 | 0.5086 | 383 |
| **Macro Average** | **0.8179** | **0.7058** | **0.7430** | **16,136** |
| **Weighted Average** | **0.8537** | **0.8547** | **0.8528** | **16,136** |

### Confusion Matrix Matrix (Holdout Test Set)

| Ground Truth \ Predicted | Entire home/apt | Private room | Shared room | Total Actual |
| :--- | :--- | :--- | :--- | :--- |
| **Entire home/apt** | 7,322 | 1,058 | 5 | 8,385 |
| **Private room** | 1,001 | 6,321 | 46 | 7,368 |
| **Shared room** | 35 | 200 | 148 | 383 |
| **Total Predicted** | 8,358 | 7,579 | 199 | 16,136 |

> **Key Takeaway**: Cost-sensitive weighting achieved a precision of **74.37%** on the extreme minority class (`Shared room`) while preserving high precision and recall (>83%–87%) across mainstream listing types.

---

## 6. API Architecture (FastAPI)

The inference service is exposed via an asynchronous REST API created with FastAPI.

### Endpoint Specifications

| HTTP Method | Endpoint | Description | Request Payload | Response Schema | Status Code |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Liveness / Health check probe | None | `{"Greet": "string"}` | `200 OK` |
| `POST` | `/predict` | Ingests listing attributes; returns classification & probabilities | JSON (`features` schema) | JSON (`Predicted_room_type`, `Probability`) | `200 OK` / `422 Unprocessable` |
| `GET` | `/docs` | Interactive Swagger UI documentation | None | HTML / JavaScript OpenAPI interface | `200 OK` |
| `GET` | `/redoc` | Interactive ReDoc API specification | None | HTML / JavaScript ReDoc interface | `200 OK` |
| `GET` | `/openapi.json` | OpenAPI 3.1.0 schema definition | None | JSON schema specification | `200 OK` |

### Ingestion Schema Contract (`features`)

```json
{
  "latitude": 40.75362,
  "longitude": -73.98377,
  "price": 225.0,
  "minimum_nights": 1,
  "number_of_reviews": 45,
  "reviews_per_month": 0.38,
  "calculated_host_listings_count": 2,
  "availability_365": 355,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}
```

#### Field Validation Rules
- `latitude`: `float` (\(-90 \le x \le 90\))
- `longitude`: `float` (\(-180 \le x \le 180\))
- `price`: `float` (\(x > 0\))
- `minimum_nights`: `integer` (\(1 \le x \le 365\))
- `number_of_reviews`: `integer` (\(x \ge 0\))
- `reviews_per_month`: `float` (\(x \ge 0.0\))
- `calculated_host_listings_count`: `integer` (\(x \ge 0\))
- `availability_365`: `integer` (\(0 \le x \le 365\))
- `neighbourhood_group`: `string` (min length: 1)
- `neighbourhood`: `string` (min length: 1)

### Prediction Response Schema

```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [
    0.8266666666666667,
    0.16666666666666666,
    0.006666666666666667
  ]
}
```

*Note: The probability vector maps strictly to class indices `[0: 'Entire home/apt', 1: 'Private room', 2: 'Shared room']`.*

---

## 7. Tech Stack

- **Runtime Environment**: Python 3.10+
- **REST Framework**: FastAPI (`v0.142.2`), Starlette (`v1.7.0`)
- **Data Validation & Serializers**: Pydantic (`v2.13.5`), Pydantic Core (`v2.46.5`)
- **ASGI Server Engine**: Uvicorn (`v0.54.0`)
- **Machine Learning & Preprocessing**: Scikit-Learn (`v1.9.1`)
- **Data Manipulation & Scientific Computing**: Pandas (`v3.0.6`), NumPy (`v2.5.3`), SciPy (`v1.18.1`)
- **Model Persistence & Serialization**: Joblib (`v1.6.0`)
- **Asynchronous Execution**: AnyIO (`v4.15.1`)

---

## 8. Local Setup & Execution

Follow these step-by-step instructions to set up the environment, install backend dependencies, and run the FastAPI inference server.

### 1. Navigate to the Backend Workspace
```bash
cd Backend
```

### 2. Configure Virtual Environment

* **On Windows (PowerShell / Command Prompt):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

* **On macOS / Linux (Bash / Zsh):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Backend Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Model Artifact
Ensure the serialized pipeline `Model_Pipeline.pkl` exists in the `Backend` directory:
```bash
# Windows PowerShell
Test-Path Model_Pipeline.pkl

# macOS / Linux
ls -lh Model_Pipeline.pkl
```

### 5. Launch the FastAPI ASGI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The service will bind to `http://127.0.0.1:8000`.

### 6. Verify Service Operation

* **Health Check Probe:**
  ```bash
  curl -X GET http://127.0.0.1:8000/
  ```
  Expected Response:
  ```json
  {"Greet": "Hello users!"}
  ```

* **Perform Inference Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "latitude": 40.75362,
      "longitude": -73.98377,
      "price": 225.0,
      "minimum_nights": 1,
      "number_of_reviews": 45,
      "reviews_per_month": 0.38,
      "calculated_host_listings_count": 2,
      "availability_365": 355,
      "neighbourhood_group": "Manhattan",
      "neighbourhood": "Midtown"
    }'
  ```

* **Interactive OpenAPI Documentation:**
  Open your web browser and navigate to:
  - **Swagger UI:** `http://127.0.0.1:8000/docs`
  - **ReDoc UI:** `http://127.0.0.1:8000/redoc`
