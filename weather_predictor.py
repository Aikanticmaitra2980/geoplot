import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class WeatherPredictorGPR:
    def __init__(self, length_scale=0.5, noise_level=0.1):
        """
        Initialize the Gaussian Process Regressor weather predictor.
        
        Parameters:
        - length_scale: Initial length scale of the RBF kernel in degrees (1 degree ≈ 111 km).
                       Determines how far the weather station's influence extends.
        - noise_level: Initial noise level representing sensor error / microclimatic noise.
        """
        # We combine RBF (smooth spatial correlation) with WhiteKernel (sensor noise)
        # We set bounds so the model can optimize these parameters during fitting.
        self.kernel = (
            1.0 * RBF(length_scale=length_scale, length_scale_bounds=(0.01, 10.0)) + 
            WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 1.0))
        )
        self.model = GaussianProcessRegressor(
            kernel=self.kernel, 
            n_restarts_optimizer=5, 
            random_state=42
        )
        self.is_fitted = False

    def fit(self, lats, lons, values):
        """
        Train the GPR model on known weather station data.
        
        Parameters:
        - lats: array-like of shape (n_samples,), latitude coordinates of stations
        - lons: array-like of shape (n_samples,), longitude coordinates of stations
        - values: array-like of shape (n_samples,), observed weather variable (e.g., temperature)
        """
        X = np.vstack([lats, lons]).T
        y = np.array(values)
        
        if len(X) < 2:
            raise ValueError("Need at least 2 weather stations to perform GPR interpolation.")
            
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Log the optimized kernel parameters
        print(f"Optimized GPR Kernel: {self.model.kernel_}")
        return self

    def predict_grid(self, min_lat, min_lon, max_lat, max_lon, grid_size=30):
        """
        Predict weather values and uncertainties across a regular coordinate grid.
        
        Parameters:
        - min_lat, min_lon, max_lat, max_lon: Viewport bounding box coordinates.
        - grid_size: Resolution of the grid (number of points along each axis).
        
        Returns:
        - A dictionary formatted as a GeoJSON FeatureCollection.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted with station data before predicting.")
            
        # Generate grid coordinates
        lat_steps = np.linspace(min_lat, max_lat, grid_size)
        lon_steps = np.linspace(min_lon, max_lon, grid_size)
        
        lat_grid, lon_grid = np.meshgrid(lat_steps, lon_steps)
        grid_points = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
        
        # Predict mean and standard deviation (uncertainty)
        y_mean, y_std = self.model.predict(grid_points, return_std=True)
        
        # Convert back to standard Python types for JSON serialization
        y_mean = y_mean.tolist()
        y_std = y_std.tolist()
        
        # Calculate cell bounds to build a grid of Polygons instead of just points.
        # This makes styling much easier in Mapbox!
        step_lat = (max_lat - min_lat) / (grid_size - 1)
        step_lon = (max_lon - min_lon) / (grid_size - 1)
        half_step_lat = step_lat / 2.0
        half_step_lon = step_lon / 2.0
        
        features = []
        for idx, (lat, lon) in enumerate(grid_points):
            val = y_mean[idx]
            unc = y_std[idx]
            
            # Construct a small rectangular polygon centered at (lat, lon)
            coords = [
                [lon - half_step_lon, lat - half_step_lat],
                [lon + half_step_lon, lat - half_step_lat],
                [lon + half_step_lon, lat + half_step_lat],
                [lon - half_step_lon, lat + half_step_lat],
                [lon - half_step_lon, lat - half_step_lat] # Close polygon
            ]
            
            features.append({
                "type": "Feature",
                "properties": {
                    "id": idx,
                    "value": float(val),
                    "uncertainty": float(unc)
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            })
            
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return geojson
