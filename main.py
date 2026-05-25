from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import httpx
import asyncio
import random

# Import our GPR predictor
from weather_predictor import WeatherPredictorGPR

app = FastAPI(
    title="Gaussian Process Weather Interpolation API",
    description="FastAPI backend serving GPR spatial interpolation layers using Open-Meteo and OpenStreetMap geodata.",
    version="1.4.0"
)

# Helper to dynamically read the token from the .env file in python
def load_env_token() -> Optional[str]:
    # Check system environment variables first
    token = os.getenv("MAPBOX_ACCESS_TOKEN") or os.getenv("MAPBOX_TOKEN")
    if token:
        return token
        
    # Check local .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        key, val = line.strip().split("=", 1)
                        if "MAPBOX" in key.upper() or "TOKEN" in key.upper():
                            return val.strip().strip('"').strip("'")
        except Exception as e:
            print(f"Error reading .env file: {e}")
    return None

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/config")
async def get_config():
    """
    Exposes configuration properties (like the Mapbox Access Token) from the server's .env file.
    """
    token = load_env_token()
    return {
        "mapbox_token": token
    }

# Request schemas
class InitializeRequest(BaseModel):
    lat: float
    lon: float

class InterpolationRequest(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    variable: str = "temp" # Supports: "temp", "rain", "clouds"
    grid_size: Optional[int] = 30

# State management: Holds the dynamically fetched weather stations
ACTIVE_STATIONS = []

# Fetch weather variables from Open-Meteo (No API Key Required!)
async def fetch_open_meteo_weather(client: httpx.AsyncClient, name: str, lat: float, lon: float) -> dict:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation,cloud_cover"
    try:
        response = await client.get(url, timeout=4.0)
        data = response.json()
        current = data["current"]
        
        return {
            "name": name,
            "lat": lat,
            "lon": lon,
            "temp": float(current["temperature_2m"]),
            "rain": float(current["precipitation"]),
            "clouds": float(current["cloud_cover"])
        }
    except Exception as e:
        print(f"Open-Meteo fetch failed for {name} ({lat}, {lon}): {e}")
        # Elegant fallback: if API fails or offline, generate realistic variables
        base_temp = 25.0 - abs(lat) * 0.2
        mock_temp = round(base_temp + random.uniform(-4.0, 4.0), 1)
        mock_rain = round(max(0.0, random.uniform(-2.0, 5.0)), 1)
        mock_clouds = round(random.uniform(0.0, 100.0), 1)
        
        return {
            "name": f"{name} (Mocked)",
            "lat": lat,
            "lon": lon,
            "temp": mock_temp,
            "rain": mock_rain,
            "clouds": mock_clouds
        }

@app.post("/api/initialize")
async def initialize_stations(req: InitializeRequest):
    """
    Finds real neighboring cities/towns in a massive regional view around the geocoded coordinates,
    fetches their real-time temperatures, precipitation, and cloud cover from Open-Meteo.
    """
    global ACTIVE_STATIONS
    
    center_lat = req.lat
    center_lon = req.lon
    
    min_lat = center_lat - 1.2
    max_lat = center_lat + 1.2
    min_lon = center_lon - 1.2
    max_lon = center_lon + 1.2
    
    nominatim_url = f"https://nominatim.openstreetmap.org/search?format=json&q=city&bounded=1&viewbox={min_lon},{max_lat},{max_lon},{min_lat}"
    headers = {"User-Agent": "AetherCastWeatherApp/1.0 (contact@aethercast.com)"}
    
    places = []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(nominatim_url, headers=headers, timeout=6.0)
            if resp.status_code == 200:
                results = resp.json()
                
                # Extract and filter major cities
                for place in results:
                    name = place.get("name")
                    lat_str = place.get("lat")
                    lon_str = place.get("lon")
                    
                    if name and lat_str and lon_str:
                        lat = float(lat_str)
                        lon = float(lon_str)
                        
                        if not any(abs(p["lat"] - lat) < 0.2 and abs(p["lon"] - lon) < 0.2 for p in places):
                            places.append({"name": name, "lat": lat, "lon": lon})
                            
                # If we don't have enough major cities, back-fill with towns
                if len(places) < 6:
                    towns_url = f"https://nominatim.openstreetmap.org/search?format=json&q=town&bounded=1&viewbox={min_lon},{max_lat},{max_lon},{min_lat}"
                    towns_resp = await client.get(towns_url, headers=headers, timeout=6.0)
                    if towns_resp.status_code == 200:
                        towns_results = towns_resp.json()
                        for place in towns_results:
                            name = place.get("name")
                            lat_str = place.get("lat")
                            lon_str = place.get("lon")
                            
                            if name and lat_str and lon_str:
                                lat = float(lat_str)
                                lon = float(lon_str)
                                
                                if not any(abs(p["lat"] - lat) < 0.15 and abs(p["lon"] - lon) < 0.15 for p in places):
                                    places.append({"name": name, "lat": lat, "lon": lon})
                                    
                            if len(places) >= 8:
                                break
    except Exception as e:
        print(f"OSM Nominatim fetch failed, using fallback dynamic offsets: {e}")

    # Fallback coordinates if OSM returns nothing
    if len(places) < 4:
        fallback_offsets = [
            (0.0, 0.0, "Center City"),
            (0.6, 0.4, "Northeast Hub"),
            (-0.5, 0.6, "Southeast Hub"),
            (-0.6, -0.5, "Southwest Hub"),
            (0.4, -0.6, "Northwest Hub"),
            (0.0, 0.8, "East Region Station"),
            (-0.8, 0.0, "West Region Station"),
            (0.7, 0.7, "Outer Border Station")
        ]
        places = []
        for d_lat, d_lon, name in fallback_offsets:
            places.append({
                "name": name,
                "lat": center_lat + d_lat,
                "lon": center_lon + d_lon
            })
            
    places = places[:8]

    # Fetch weather variables from Open-Meteo in parallel
    async with httpx.AsyncClient() as client:
        tasks = []
        for place in places:
            tasks.append(fetch_open_meteo_weather(client, place["name"], place["lat"], place["lon"]))
            
        results = await asyncio.gather(*tasks)
        
    ACTIVE_STATIONS = results
    return await get_stations()

@app.get("/api/stations")
async def get_stations():
    """
    Returns currently active weather stations with all variables in GeoJSON format.
    """
    if not ACTIVE_STATIONS:
        return {
            "type": "FeatureCollection",
            "features": []
        }
        
    features = []
    for idx, station in enumerate(ACTIVE_STATIONS):
        features.append({
            "type": "Feature",
            "properties": {
                "id": idx,
                "name": station["name"],
                "temperature": station["temp"],
                "rain": station["rain"],
                "clouds": station["clouds"]
            },
            "geometry": {
                "type": "Point",
                "coordinates": [station["lon"], station["lat"]]
            }
        })
        
    return {
        "type": "FeatureCollection",
        "features": features
    }

@app.post("/api/interpolate")
async def interpolate_weather(req: InterpolationRequest):
    """
    Fits GPR on selected weather variable (temp, rain, clouds) and interpolates values over the grid viewport.
    """
    try:
        if not ACTIVE_STATIONS:
            return {"type": "FeatureCollection", "features": []}
            
        lats = [s["lat"] for s in ACTIVE_STATIONS]
        lons = [s["lon"] for s in ACTIVE_STATIONS]
        
        # Select target variable for GPR mapping
        if req.variable == "rain":
            values = [s["rain"] for s in ACTIVE_STATIONS]
        elif req.variable == "clouds":
            values = [s["clouds"] for s in ACTIVE_STATIONS]
        else: # "temp"
            values = [s["temp"] for s in ACTIVE_STATIONS]
        
        # Auto-adjust GPR length_scale
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        scale = max(0.05, (lat_range + lon_range) / 4.0)
        
        # Fit GPR Model
        predictor = WeatherPredictorGPR(length_scale=scale, noise_level=0.01)
        predictor.fit(lats, lons, values)
        
        # Predict over viewport
        geojson_grid = predictor.predict_grid(
            min_lat=req.min_lat,
            min_lon=req.min_lon,
            max_lat=req.max_lat,
            max_lon=req.max_lon,
            grid_size=req.grid_size
        )
        
        return geojson_grid
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    
    return """
    <html>
        <head><title>FastAPI Interpolation Server</title></head>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
            <h1>FastAPI Weather Server is Running!</h1>
            <p>Please ensure index.html exists in the directory.</p>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
