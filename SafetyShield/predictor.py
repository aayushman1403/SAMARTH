import numpy as np
import logging
import random
import datetime
from pathlib import Path
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import joblib
import os

logger = logging.getLogger(__name__)

class SafetyPredictor:
    """
    Predicts safety levels for Mumbai wards based on time of day.
    Uses a hybrid model approach combining:
    - LSTM for time-series analysis
    - Random Forest + XGBoost for spatio-temporal predictions
    - Historical pattern recognition for trend analysis
    """
    
    def __init__(self):
        """Initialize the safety predictor."""
        self.rf_model = None
        self.gb_model = None
        self.hybrid_model_path = Path("static/models/hybrid_model.joblib")
        self.lstm_model_path = Path("static/models/lstm_model.joblib")
        self.wards = None
        self.ward_geojson_path = Path("static/data/mumbai_wards.geojson")
        self.historical_data = {}  # Store historical predictions for each ward
        
        # Create models directory if it doesn't exist
        os.makedirs("static/models", exist_ok=True)
        
        # Load ward data
        try:
            self._load_ward_data()
            self._initialize_models()
            self._generate_historical_data()
        except Exception as e:
            logger.error(f"Error initializing SafetyPredictor: {str(e)}")
    
    def _load_ward_data(self):
        """Load ward data from GeoJSON file."""
        try:
            if self.ward_geojson_path.exists():
                with open(self.ward_geojson_path, 'r') as f:
                    self.wards = json.load(f)
                logger.info(f"Loaded {len(self.wards['features'])} ward boundaries for prediction")
            else:
                logger.warning(f"Ward GeoJSON file not found at {self.ward_geojson_path}")
                # Create a placeholder for development
                self.wards = {"type": "FeatureCollection", "features": []}
                
                # Create 24 placeholder wards for demonstration
                for i in range(1, 25):
                    ward_id = f"W{i:02d}"
                    ward_feature = {
                        "type": "Feature",
                        "properties": {
                            "ward_id": ward_id,
                            "name": f"Ward {i}",
                            "population": random.randint(50000, 200000)
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[]]  # Empty coordinates
                        }
                    }
                    self.wards["features"].append(ward_feature)
                logger.info("Created placeholder ward data for demonstration")
        except Exception as e:
            logger.error(f"Error loading ward data: {str(e)}")
            self.wards = {"type": "FeatureCollection", "features": []}
    
    def _load_model(self):
        """
        Load the ConvLSTM model for prediction.
        In a real implementation, this would load a trained Keras model.
        """
        logger.info("Model loading is simulated for demonstration")
        # Simulate model loading - in a real implementation, this would use:
        # self.model = tf.keras.models.load_model('path/to/model')
        
        # For demonstration, we'll just set a flag
        self.model = True
        return True
    
    def predict(self, hour):
        """
        Predict safety levels for all wards based on the hour of day.
        
        Args:
            hour (int): Hour of the day (0-23)
            
        Returns:
            dict: Prediction results including ward safety levels
        """
        if not self.model:
            self._load_model()
        
        if not self.wards or len(self.wards["features"]) == 0:
            logger.warning("No ward data available for prediction")
            return {"error": "No ward data available"}
        
        # In a real implementation, this would use the ConvLSTM model
        # For demonstration, we'll create a simplified safety level prediction
        
        # These patterns simulate the expected safety patterns throughout the day
        time_patterns = {
            "safest": [10, 11, 12, 13, 14, 15],  # Safest during mid-day
            "safe": [7, 8, 9, 16, 17, 18],       # Safer during morning/evening
            "moderate": [19, 20, 6],             # Moderate risk during early night/early morning
            "risky": [21, 22, 5],                # Higher risk in late evening/early morning
            "high_risk": [23, 0, 1, 2, 3, 4]     # Most dangerous during late night/early morning hours
        }
        
        # Calculate base safety probabilities based on the hour
        if hour in time_patterns["safest"]:
            # Mid-day hours are safest
            base_safety = 0.8 + (0.1 * np.sin(np.pi * hour / 12))
        elif hour in time_patterns["safe"]:
            # Morning/evening are generally safe
            base_safety = 0.7 + (0.1 * np.sin(np.pi * hour / 12))
        elif hour in time_patterns["moderate"]:
            # Transition hours have moderate safety
            base_safety = 0.5 + (0.1 * np.sin(np.pi * hour / 12))
        elif hour in time_patterns["risky"]:
            # Late evening/early morning are riskier
            base_safety = 0.3 + (0.1 * np.sin(np.pi * hour / 12))
        else:
            # Late night hours are least safe
            base_safety = 0.2 + (0.1 * np.sin(np.pi * hour / 12))
        
        # Create predictions for each ward
        predictions = {
            "hour": hour,
            "timestamp": f"{hour:02d}:00",
            "wards": []
        }
        
        # Generate predictions for each ward
        for feature in self.wards["features"]:
            ward_id = feature["properties"].get("ward_id", "unknown")
            name = feature["properties"].get("name", f"Ward {ward_id}")
            
            # Adjust safety by ward (simplified simulation)
            # In reality, this would come from the model prediction
            ward_modifier = hash(ward_id) % 20 / 100  # -0.1 to 0.1 variation
            safety_score = min(max(base_safety + ward_modifier, 0), 1)
            
            # Determine safety level
            if safety_score >= 0.7:
                safety_level = "green"
            elif safety_score >= 0.4:
                safety_level = "yellow"
            else:
                safety_level = "red"
            
            # Calculate a crime probability (inverse of safety)
            crime_probability = 1 - safety_score
            
            ward_prediction = {
                "ward_id": ward_id,
                "name": name,
                "safety_level": safety_level,
                "crime_probability": round(crime_probability, 3),
                "risk_factors": self._get_risk_factors(hour, ward_id)
            }
            
            predictions["wards"].append(ward_prediction)
        
        return predictions
    
    def _initialize_models(self):
        """Initialize the RF+GB hybrid model and LSTM model for time-series prediction"""
        try:
            # For a real implementation, we would load pre-trained models
            # If models don't exist, train simple models for demonstration
            
            if not self.hybrid_model_path.exists():
                logger.info("Training demonstration hybrid model")
                # Initialize a simple random forest model
                self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
                self.gb_model = GradientBoostingClassifier(n_estimators=10, random_state=42)
                
                # For demonstration, we're just initializing the model
                # In a real implementation, this would train on actual crime data
                X_train = np.random.rand(100, 5)  # Features: hour, day, month, latitude, longitude
                y_train = np.random.randint(0, 3, 100)  # 0=safe, 1=moderate, 2=dangerous
                
                # Train models
                self.rf_model.fit(X_train, y_train)
                self.gb_model.fit(X_train, y_train)
                
                # Save models for future use
                joblib.dump((self.rf_model, self.gb_model), self.hybrid_model_path)
            else:
                # Load pre-trained models
                logger.info("Loading hybrid model")
                self.rf_model, self.gb_model = joblib.load(self.hybrid_model_path)
                
            # Set model flag for prediction function
            self.model = True
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            self.model = False

    def _generate_historical_data(self):
        """Generate historical safety data for the past 30 days for each ward"""
        try:
            today = datetime.datetime.now()
            wards = [feature["properties"].get("ward_id") for feature in self.wards["features"]]
            
            for ward_id in wards:
                # Generate 30 days of hourly data for each ward
                daily_data = []
                
                for day_offset in range(30, 0, -1):
                    date = today - datetime.timedelta(days=day_offset)
                    
                    # Generate 24 hours of data for each day
                    hourly_data = []
                    for hour in range(24):
                        # Base safety score depends on hour with more variation by time periods
                        # We add some random variation by day and ward
                        day_factor = hash(f"{date.weekday()}_{ward_id}") % 20 / 100
                        
                        # Use the same time patterns as in predict function for consistency
                        if hour in [10, 11, 12, 13, 14, 15]:
                            # Mid-day hours are safest
                            hour_factor = 0.8
                        elif hour in [7, 8, 9, 16, 17, 18]:
                            # Morning/evening are generally safe
                            hour_factor = 0.7
                        elif hour in [19, 20, 6]:
                            # Transition hours have moderate safety
                            hour_factor = 0.5
                        elif hour in [21, 22, 5]:
                            # Late evening/early morning are riskier
                            hour_factor = 0.3
                        else:
                            # Late night hours (11pm-4am) are least safe
                            hour_factor = 0.2
                        
                        safety_score = min(max(hour_factor + day_factor, 0), 1)
                        
                        # Convert to safety level
                        if safety_score >= 0.7:
                            safety_level = "green"
                        elif safety_score >= 0.4:
                            safety_level = "yellow"
                        else:
                            safety_level = "red"
                        
                        hourly_data.append({
                            "hour": hour,
                            "safety_level": safety_level,
                            "crime_probability": round(1 - safety_score, 3)
                        })
                    
                    daily_data.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "weekday": date.strftime("%A"),
                        "hourly_data": hourly_data
                    })
                
                self.historical_data[ward_id] = daily_data
            
            logger.info(f"Generated historical data for {len(wards)} wards over 30 days")
        
        except Exception as e:
            logger.error(f"Error generating historical data: {str(e)}")
            self.historical_data = {}

    def get_historical_safety_data(self, ward_id, days=7):
        """
        Get historical safety data for a specific ward
        
        Args:
            ward_id (str): Ward identifier
            days (int): Number of days of history to return (default 7, max 30)
            
        Returns:
            dict: Historical safety data for the ward
        """
        days = min(days, 30)  # Limit to 30 days maximum
        
        if ward_id not in self.historical_data:
            return {"error": f"No historical data for ward {ward_id}"}
        
        historical_data = self.historical_data[ward_id][-days:]  # Last 'days' entries
        
        # Calculate average safety levels across different time periods
        time_periods = {
            "morning": [6, 7, 8, 9, 10, 11],
            "afternoon": [12, 13, 14, 15, 16, 17],
            "evening": [18, 19, 20, 21],
            "night": [22, 23, 0, 1, 2, 3, 4, 5]
        }
        
        period_stats = {}
        for period, hours in time_periods.items():
            green_count = yellow_count = red_count = 0
            total_hours = len(hours) * days
            
            for day_data in historical_data:
                for hour_data in day_data["hourly_data"]:
                    if hour_data["hour"] in hours:
                        if hour_data["safety_level"] == "green":
                            green_count += 1
                        elif hour_data["safety_level"] == "yellow":
                            yellow_count += 1
                        else:
                            red_count += 1
            
            # Determine dominant safety level
            if green_count >= yellow_count and green_count >= red_count:
                dominant = "green"
                dominant_pct = (green_count / total_hours) * 100
            elif yellow_count >= green_count and yellow_count >= red_count:
                dominant = "yellow"
                dominant_pct = (yellow_count / total_hours) * 100
            else:
                dominant = "red"
                dominant_pct = (red_count / total_hours) * 100
            
            period_stats[period] = {
                "dominant_safety": dominant,
                "dominant_percentage": round(dominant_pct, 1),
                "green_pct": round((green_count / total_hours) * 100, 1),
                "yellow_pct": round((yellow_count / total_hours) * 100, 1),
                "red_pct": round((red_count / total_hours) * 100, 1)
            }
        
        return {
            "ward_id": ward_id,
            "days_analyzed": days,
            "daily_data": historical_data,
            "period_stats": period_stats
        }

    def predict_future_risk(self, ward_id, future_hours=24):
        """
        Predict future risk levels for a specific ward
        
        Args:
            ward_id (str): Ward identifier
            future_hours (int): Number of hours to predict into the future
            
        Returns:
            dict: Predicted future risk levels
        """
        if not self.rf_model or not self.gb_model:
            return {"error": "Prediction models not available"}
        
        # Get ward information
        ward_info = None
        for feature in self.wards["features"]:
            if feature["properties"].get("ward_id") == ward_id:
                ward_info = feature
                break
        
        if not ward_info:
            return {"error": f"Ward {ward_id} not found"}
        
        # Current time
        now = datetime.datetime.now()
        
        # Generate predictions for future hours
        predictions = []
        for i in range(future_hours):
            future_time = now + datetime.timedelta(hours=i)
            hour = future_time.hour
            day = future_time.weekday()  # 0=Monday, 6=Sunday
            
            # Extract coordinates from ward_info - using center point for simplicity
            try:
                coords = ward_info["geometry"]["coordinates"][0]
                if coords:
                    # Calculate centroid for polygon (simplified)
                    lats = [coord[1] for coord in coords if len(coord) >= 2]
                    longs = [coord[0] for coord in coords if len(coord) >= 2]
                    if lats and longs:
                        lat = sum(lats) / len(lats)
                        lng = sum(longs) / len(longs)
                    else:
                        # Use Mumbai coordinates as fallback
                        lat, lng = 19.076, 72.8777
                else:
                    # Use Mumbai coordinates as fallback
                    lat, lng = 19.076, 72.8777
            except (IndexError, KeyError):
                # Use Mumbai coordinates as fallback
                lat, lng = 19.076, 72.8777
            
            # Prepare features for prediction
            features = np.array([[hour, day, future_time.month, lat, lng]])
            
            # Make predictions with both models
            rf_pred = self.rf_model.predict_proba(features)[0]
            gb_pred = self.gb_model.predict_proba(features)[0]
            
            # Ensemble the predictions (average)
            ensemble_pred = (rf_pred + gb_pred) / 2
            
            # Get the class with highest probability
            risk_class = np.argmax(ensemble_pred)
            
            # Convert to safety level
            if risk_class == 0:
                safety_level = "green"
            elif risk_class == 1:
                safety_level = "yellow"
            else:
                safety_level = "red"
            
            # Add prediction to results
            predictions.append({
                "timestamp": future_time.strftime("%Y-%m-%d %H:00"),
                "hour": hour,
                "safety_level": safety_level,
                "probability": round(ensemble_pred[risk_class], 3),
                "risk_factors": self._get_risk_factors(hour, ward_id)
            })
        
        return {
            "ward_id": ward_id,
            "ward_name": ward_info["properties"].get("name", f"Ward {ward_id}"),
            "predictions": predictions
        }

    def get_safety_tips(self, ward_id, hour=None):
        """
        Get safety tips specific to a ward and time of day
        
        Args:
            ward_id (str): Ward identifier
            hour (int): Hour of the day (0-23), defaults to current hour
            
        Returns:
            dict: Safety tips for the specified ward
        """
        if hour is None:
            hour = datetime.datetime.now().hour
            
        # Get the safety level for this ward and hour
        safety_data = self.predict(hour)
        ward_data = None
        
        for ward in safety_data.get("wards", []):
            if ward["ward_id"] == ward_id:
                ward_data = ward
                break
                
        if not ward_data:
            return {"error": f"No data available for ward {ward_id}"}
        
        safety_level = ward_data["safety_level"]
        risk_factors = ward_data["risk_factors"]
        
        # General safety tips
        general_tips = [
            "Stay aware of your surroundings at all times",
            "Keep emergency contacts readily available",
            "Share your location with trusted contacts when traveling",
            "Stay in well-lit and populated areas when possible"
        ]
        
        # Safety tips based on safety level
        level_tips = {
            "green": [
                "This area is generally safe, but basic precautions are still recommended",
                "Normal vigilance is sufficient in this area",
                "Enjoy your activities while maintaining standard awareness"
            ],
            "yellow": [
                "Moderate caution is advised in this area",
                "Avoid walking alone at night if possible",
                "Keep valuables concealed and secure",
                "Stay in well-lit and populated areas"
            ],
            "red": [
                "Extra vigilance is strongly recommended",
                "Avoid traveling alone, especially after dark",
                "Consider alternative routes or transportation",
                "Keep in constant contact with someone who knows your whereabouts",
                "Avoid displaying valuable items in public"
            ]
        }
        
        # Risk factor specific tips
        factor_tips = {
            "Poorly lit areas": [
                "Use a flashlight or phone light in dark areas",
                "Stick to main roads with proper lighting",
                "Travel in groups when possible"
            ],
            "High pedestrian traffic": [
                "Keep your wallet/purse secure and close to your body",
                "Be aware of pickpockets in crowded areas",
                "Avoid distractions like using phone in very crowded places"
            ],
            "Proximity to transit hubs": [
                "Be extra vigilant around bus and train stations",
                "Secure your luggage and personal belongings",
                "Pre-plan your route to minimize waiting time"
            ],
            "Entertainment venues": [
                "Consume alcohol responsibly if visiting bars/clubs",
                "Never leave drinks unattended",
                "Plan your return transportation in advance"
            ],
            "Commercial activity": [
                "Keep shopping bags close and monitored",
                "Avoid displaying large amounts of cash",
                "Be cautious in market areas with dense crowds"
            ],
            "Residential density": [
                "Check if your building has functioning security measures",
                "Lock doors and windows properly",
                "Be aware of your neighbors and report suspicious activity"
            ],
            "Previous incidents": [
                "Check local news for recent crime patterns in this area",
                "Avoid areas with repeated criminal activity",
                "Follow police advisories for this location"
            ],
            "School/college proximity": [
                "Be alert during opening and closing hours when crowds gather",
                "Watch for traffic congestion during school rush hours",
                "Report suspicious individuals loitering near educational institutions"
            ]
        }
        
        # Compile the tips
        specific_tips = level_tips.get(safety_level, [])
        
        # Add factor-specific tips
        for factor in risk_factors:
            if factor in factor_tips:
                specific_tips.extend(factor_tips[factor])
        
        # Time-specific tips
        time_tips = []
        if 0 <= hour < 6:  # Late night
            time_tips = [
                "Consider using private transportation rather than walking",
                "Inform someone of your expected arrival time",
                "Avoid poorly lit shortcuts"
            ]
        elif 6 <= hour < 12:  # Morning
            time_tips = [
                "Morning rush hour may create opportunities for pickpockets",
                "Be cautious at ATMs during early banking hours",
                "Watch for traffic congestion around schools and offices"
            ]
        elif 12 <= hour < 18:  # Afternoon
            time_tips = [
                "Be cautious in crowded shopping areas during peak hours",
                "Stay hydrated and watch for heat-related health issues",
                "Be alert for potential scams in tourist areas"
            ]
        else:  # Evening
            time_tips = [
                "Prefer well-traveled routes after sunset",
                "Keep your phone charged for emergencies",
                "Avoid displaying valuable electronics in public"
            ]
        
        return {
            "ward_id": ward_id,
            "ward_name": ward_data["name"],
            "safety_level": safety_level,
            "general_tips": general_tips,
            "specific_tips": specific_tips,
            "time_tips": time_tips
        }

    def _get_risk_factors(self, hour, ward_id):
        """
        Generate risk factors for a specific ward at the given hour.
        This is a simplified placeholder.
        
        Args:
            hour (int): Hour of the day
            ward_id (str): Ward identifier
            
        Returns:
            list: Risk factors affecting safety
        """
        # Simplified risk factors based on time of day
        all_factors = [
            "Poorly lit areas",
            "High pedestrian traffic",
            "Proximity to transit hubs",
            "Entertainment venues",
            "Commercial activity",
            "Residential density",
            "Previous incidents",
            "School/college proximity"
        ]
        
        # Select a subset of factors based on ward_id and hour
        seed = hash(f"{ward_id}_{hour}")
        random.seed(seed)
        num_factors = random.randint(1, 3)
        factors = random.sample(all_factors, num_factors)
        
        return factors
