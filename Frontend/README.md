# Room type predictor (frontend)

React + Vite + Tailwind CSS (v3) frontend for the Airbnb room type FastAPI backend.

## Run

    npm install
    npm run dev

Open http://localhost:5173

## Backend

The app posts to `http://localhost:8000/predict`. To change it, edit `API_URL` at the top of `src/App.jsx`.

Enable CORS in FastAPI:

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

## Class labels

`CLASS_LABELS` in `src/App.jsx` maps the probability array to room types.
Reorder it if your model's `classes_` differ from
["Entire home/apt", "Private room", "Shared room"].
