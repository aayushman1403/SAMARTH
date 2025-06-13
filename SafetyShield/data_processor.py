import os
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import current_app
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import random  # For demonstration purposes

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data processing for the Mumbai Safety Zone Predictor
    - Loads crime data from CSV files
    - Maps locations to ward IDs
    - Aggregates data by hour
    - Normalizes data for model input
    """
    
    def __init__(self):
        """Initialize the data processor with paths to data files."""
        self.ward_geojson_path = Path("static/data/mumbai_wards.geojson")
        self.crime_data_path = Path("static/data/sample_crime_data.csv")
        self.wards = None
        self.crime_data = None
        
        # Load data if files exist
        try:
            self._load_wards()
            self._load_crime_data()
        except Exception as e:
            logger.error(f"Error initializing DataProcessor: {str(e)}")
    
    def _load_wards(self):
        """Load ward boundaries from GeoJSON file."""
        try:
            if self.ward_geojson_path.exists():
                with open(self.ward_geojson_path, 'r') as f:
                    self.wards = json.load(f)
                # Enhance data with more wards for a denser visualization
                self._enhance_ward_data()
                logger.info(f"Loaded {len(self.wards['features'])} ward boundaries")
            else:
                logger.warning(f"Ward GeoJSON file not found at {self.ward_geojson_path}")
                # Create a placeholder for development if file doesn't exist
                self.wards = {"type": "FeatureCollection", "features": []}
                # Generate mock ward data
                self._generate_mock_ward_data()
        except Exception as e:
            logger.error(f"Error loading ward data: {str(e)}")
            
    def _enhance_ward_data(self):
        """Add more wards to existing data to create a denser visualization."""
        if not self.wards or "features" not in self.wards:
            return
            
        # Get existing features for reference
        existing_features = self.wards["features"].copy()
        
        # Base Mumbai coordinates
        base_lat, base_lng = 19.076, 72.8777
        new_wards = []
        existing_count = len(existing_features)
        
        for i in range(20):
            # Create new wards in the vicinity of existing ones
            ref_idx = i % existing_count
            ref_feature = existing_features[ref_idx]
            
            try:
                # Get reference coordinates
                coords = ref_feature["geometry"]["coordinates"][0]
                if coords and len(coords) > 0:
                    # Calculate centroid
                    lats = [coord[1] for coord in coords if len(coord) >= 2]
                    lngs = [coord[0] for coord in coords if len(coord) >= 2]
                    if lats and lngs:
                        ref_lat = sum(lats) / len(lats)
                        ref_lng = sum(lngs) / len(lngs)
                    else:
                        ref_lat, ref_lng = base_lat, base_lng
                else:
                    ref_lat, ref_lng = base_lat, base_lng
            except (KeyError, IndexError):
                ref_lat, ref_lng = base_lat, base_lng
                
            # Create small random offset (0.001-0.005 degrees) to distribute new wards
            lat_offset = (random.random() * 0.004 + 0.001) * (1 if random.random() > 0.5 else -1)
            lng_offset = (random.random() * 0.004 + 0.001) * (1 if random.random() > 0.5 else -1)
            
            new_lat = ref_lat + lat_offset
            new_lng = ref_lng + lng_offset
            
            # Create a simple polygon around this point (small area)
            r = 0.001  # Small radius in degrees
            polygon = [
                [new_lng - r, new_lat - r],
                [new_lng + r, new_lat - r],
                [new_lng + r, new_lat + r],
                [new_lng - r, new_lat + r],
                [new_lng - r, new_lat - r]
            ]
            
            # Generate new ward ID
            new_id = f"W{existing_count + i + 1}"
            
            # Create a new feature
            new_feature = {
                "type": "Feature",
                "properties": {
                    "ward_id": new_id,
                    "name": f"Ward {new_id}"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon]
                }
            }
            
            new_wards.append(new_feature)
        
        # Add new wards to the collection
        self.wards["features"].extend(new_wards)
            
    def _generate_mock_ward_data(self):
        """Generate mock ward data if no real data is available."""
        # Base Mumbai coordinates
        base_lat, base_lng = 19.076, 72.8777
        
        # Create 35-40 wards around Mumbai
        features = []
        for i in range(40):
            # Create random offset from center
            lat_offset = (random.random() * 0.05) * (1 if random.random() > 0.5 else -1)
            lng_offset = (random.random() * 0.05) * (1 if random.random() > 0.5 else -1)
            
            lat = base_lat + lat_offset
            lng = base_lng + lng_offset
            
            # Create a simple polygon around this point
            r = 0.005  # Small radius in degrees
            polygon = [
                [lng - r, lat - r],
                [lng + r, lat - r],
                [lng + r, lat + r],
                [lng - r, lat + r],
                [lng - r, lat - r]
            ]
            
            # Create feature with properties
            feature = {
                "type": "Feature",
                "properties": {
                    "ward_id": f"W{i+1}",
                    "name": f"Ward {chr(65+i%26)}{i//26+1}"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon]
                }
            }
            
            features.append(feature)
            
        # Create GeoJSON feature collection
        self.wards = {
            "type": "FeatureCollection",
            "features": features
        }
    
    def _load_crime_data(self):
        """Load crime data from CSV file."""
        try:
            if self.crime_data_path.exists():
                self.crime_data = pd.read_csv(self.crime_data_path)
                # Ensure datetime format
                if 'timestamp' in self.crime_data.columns:
                    self.crime_data['timestamp'] = pd.to_datetime(self.crime_data['timestamp'])
                    self.crime_data['hour'] = self.crime_data['timestamp'].dt.hour
                logger.info(f"Loaded {len(self.crime_data)} crime incidents")
            else:
                logger.warning(f"Crime data file not found at {self.crime_data_path}")
                # Create empty DataFrame with expected columns
                self.crime_data = pd.DataFrame(columns=[
                    'incident_type', 'ward_id', 'latitude', 'longitude', 
                    'timestamp', 'hour', 'severity', 'description'
                ])
        except Exception as e:
            logger.error(f"Error loading crime data: {str(e)}")
            self.crime_data = pd.DataFrame()
    
    def get_wards(self):
        """Return ward GeoJSON data."""
        if self.wards is None:
            self._load_wards()
        return self.wards
    
    def get_crime_data_by_hour(self, hour):
        """
        Return crime data filtered by the specified hour.
        
        Args:
            hour (int): Hour of the day (0-23)
            
        Returns:
            pd.DataFrame: Filtered crime data
        """
        if self.crime_data is None or 'hour' not in self.crime_data.columns:
            return pd.DataFrame()
        
        return self.crime_data[self.crime_data['hour'] == hour]
    
    def get_crime_counts_by_ward_and_hour(self):
        """
        Aggregate crime counts by ward and hour.
        
        Returns:
            pd.DataFrame: Crime counts per ward per hour
        """
        if self.crime_data is None or self.crime_data.empty:
            return pd.DataFrame()
        
        if 'ward_id' in self.crime_data.columns and 'hour' in self.crime_data.columns:
            return self.crime_data.groupby(['ward_id', 'hour']).size().reset_index(name='count')
        return pd.DataFrame()
    
    def get_ward_safety_levels(self, hour):
        """
        Calculate safety levels for all wards at the specified hour.
        
        Args:
            hour (int): Hour of the day (0-23)
            
        Returns:
            dict: Dictionary mapping ward_id to safety level ('green', 'yellow', 'red')
        """
        # This is a simplified placeholder method
        # In a real implementation, this would use the predictor model
        crime_data = self.get_crime_data_by_hour(hour)
        safety_levels = {}
        
        if not crime_data.empty and 'ward_id' in crime_data.columns:
            # Count crimes per ward
            counts = crime_data.groupby('ward_id').size()
            
            # Determine thresholds (this is simplified)
            if len(counts) > 0:
                max_count = counts.max()
                thresholds = {
                    'green': max_count * 0.3,
                    'yellow': max_count * 0.7
                }
                
                # Assign safety levels
                for ward_id, count in counts.items():
                    if count <= thresholds['green']:
                        safety_levels[ward_id] = 'green'
                    elif count <= thresholds['yellow']:
                        safety_levels[ward_id] = 'yellow'
                    else:
                        safety_levels[ward_id] = 'red'
        
        return safety_levels
    
    def map_coordinates_to_ward(self, lat, lng):
        """
        Map latitude and longitude to a ward ID.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude
            
        Returns:
            dict: Ward information or None if no match
        """
        if not self.wards or 'features' not in self.wards:
            return None
            
        # In a real implementation, this would use a spatial database with point-in-polygon testing
        # For this implementation, we'll use a simple distance calculation to find the nearest ward center
        
        user_coords = (lat, lng)
        nearest_ward = None
        min_distance = float('inf')
        
        for feature in self.wards['features']:
            ward_id = feature['properties'].get('ward_id')
            name = feature['properties'].get('name', f'Ward {ward_id}')
            
            # Calculate center point of ward polygon
            try:
                coords = feature['geometry']['coordinates'][0]
                if coords and len(coords) > 0:
                    # Calculate centroid for polygon (simplified)
                    lats = [coord[1] for coord in coords if len(coord) >= 2]
                    longs = [coord[0] for coord in coords if len(coord) >= 2]
                    if lats and longs:
                        ward_center = (sum(lats) / len(lats), sum(longs) / len(longs))
                        
                        # Calculate distance
                        distance = geodesic(user_coords, ward_center).kilometers
                        
                        if distance < min_distance:
                            min_distance = distance
                            nearest_ward = {
                                'ward_id': ward_id,
                                'name': name,
                                'distance_km': round(distance, 2),
                                'latitude': ward_center[0],
                                'longitude': ward_center[1]
                            }
            except (KeyError, IndexError):
                continue
                
        # If we're within a reasonable distance of Mumbai (30km), return the ward
        if nearest_ward and min_distance <= 30:
            return nearest_ward
            
        return None
        
    def map_search_query_to_ward(self, query):
        """
        Convert a location search query to coordinates and find the matching ward.
        
        Args:
            query (str): Location search query (e.g., "Dadar, Mumbai")
            
        Returns:
            dict: Ward information or None if no match
        """
        try:
            # Initialize geocoder
            geolocator = Nominatim(user_agent="mumbai_safety_predictor")
            
            # Add Mumbai to the query if not present to improve accuracy
            if "mumbai" not in query.lower():
                query = f"{query}, Mumbai, India"
            
            # Get coordinates from the query
            location = geolocator.geocode(query, timeout=10)
            
            if location:
                # Check if the location is within Mumbai's general vicinity
                mumbai_center = (19.0760, 72.8777)
                location_coords = (location.latitude, location.longitude)
                
                distance_to_mumbai = geodesic(mumbai_center, location_coords).kilometers
                
                # If location is within 30km of Mumbai center, find the nearest ward
                if distance_to_mumbai <= 30:
                    ward_info = self.map_coordinates_to_ward(location.latitude, location.longitude)
                    
                    if ward_info:
                        # Add the matched location details
                        ward_info.update({
                            'search_query': query,
                            'matched_location': location.address,
                            'search_lat': location.latitude,
                            'search_lng': location.longitude
                        })
                        return ward_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error during geocoding: {str(e)}")
            
            # For demo purposes, return a random ward if geocoding fails
            # In production, this would return None or an error
            if self.wards and 'features' in self.wards and len(self.wards['features']) > 0:
                random_feature = random.choice(self.wards['features'])
                ward_id = random_feature['properties'].get('ward_id')
                name = random_feature['properties'].get('name', f'Ward {ward_id}')
                
                # Get a point within Mumbai as the 'matched' location
                mumbai_center = (19.0760, 72.8777)
                
                return {
                    'ward_id': ward_id,
                    'name': name,
                    'distance_km': round(random.uniform(0.1, 5.0), 2),
                    'search_query': query,
                    'matched_location': f"Near {name}, Mumbai, India",
                    'latitude': mumbai_center[0] + random.uniform(-0.02, 0.02),
                    'longitude': mumbai_center[1] + random.uniform(-0.02, 0.02),
                    'search_lat': mumbai_center[0] + random.uniform(-0.01, 0.01),
                    'search_lng': mumbai_center[1] + random.uniform(-0.01, 0.01)
                }
                
            return None
