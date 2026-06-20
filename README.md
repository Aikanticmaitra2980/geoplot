# Gaussian Process Weather Interpolation API

A high-performance FastAPI backend that implements real-time spatial weather interpolation using **Gaussian Process Regression (GPR)**. The system dynamically queries geospatial data from OpenStreetMap (Nominatim) and real-time weather metrics from Open-Meteo, generating a continuous geospatial prediction grid.

---

## 🚀 Features

* **Dynamic Station Discovery:** Discovers major cities and towns within a specified bounding box viewport using the OpenStreetMap Nominatim API.
* **Real-time Live Weather:** Fetches live temperature, precipitation, and cloud cover updates via the Open-Meteo API without requiring explicit developer API keys.
* **Gaussian Process Regression (GPR):** Dynamically scales and fits a spatial GPR model (`WeatherPredictorGPR`) over point data to extrapolate continuous spatial variables.
* **GeoJSON Standard Outputs:** Serves both weather stations (`Points`) and interpolation layers (`Polygons`/`Grids`) in native GeoJSON structure for seamless map rendering (Mapbox, Leaflet, etc.).
* **Robust Fail-Safes:** Includes robust async request fault tolerance with automated mock atmospheric data fallbacks in case of network drops or API rate-limiting.

---

## 🛠️ Architecture & Requirements

Ensure you are running **Python 3.8+**. You can set up the environment and download dependencies via `pip`:

```bash
pip install fastapi uvicorn httpx pydantic numpy scikit-learn
