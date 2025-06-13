import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define Base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with the Base class
db = SQLAlchemy(model_class=Base)

# Create the Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///safety_predictor.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

# Import routes after app initialization to avoid circular imports
from data_processor import DataProcessor
from predictor import SafetyPredictor

# Initialize data processor and predictor
data_processor = DataProcessor()
safety_predictor = SafetyPredictor()

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page with information about the application."""
    return render_template('about.html')

@app.route('/api/predict', methods=['GET'])
def predict():
    """API endpoint to get safety predictions based on hour of day."""
    try:
        hour = int(request.args.get('hour', 12))
        if hour < 0 or hour > 23:
            return jsonify({'error': 'Hour must be between 0 and 23'}), 400
            
        predictions = safety_predictor.predict(hour)
        return jsonify(predictions)
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/api/wards')
def get_wards():
    """API endpoint to get Mumbai ward information."""
    try:
        wards = data_processor.get_wards()
        return jsonify(wards)
    except Exception as e:
        logger.error(f"Error retrieving ward data: {str(e)}")
        return jsonify({'error': 'Failed to retrieve ward data'}), 500

@app.route('/api/historical/<ward_id>')
def get_historical_data(ward_id):
    """API endpoint to get historical safety data for a specific ward."""
    try:
        days = int(request.args.get('days', 7))
        historical_data = safety_predictor.get_historical_safety_data(ward_id, days)
        return jsonify(historical_data)
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error retrieving historical data: {str(e)}")
        return jsonify({'error': 'Failed to retrieve historical data'}), 500

@app.route('/api/future/<ward_id>')
def predict_future(ward_id):
    """API endpoint to predict future safety for a specific ward."""
    try:
        hours = int(request.args.get('hours', 24))
        future_data = safety_predictor.predict_future_risk(ward_id, hours)
        return jsonify(future_data)
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error predicting future data: {str(e)}")
        return jsonify({'error': 'Failed to generate future predictions'}), 500

@app.route('/api/tips/<ward_id>')
def get_safety_tips(ward_id):
    """API endpoint to get safety tips for a specific ward at a specific hour."""
    try:
        hour = request.args.get('hour')
        if hour is not None:
            hour = int(hour)
            if hour < 0 or hour > 23:
                return jsonify({'error': 'Hour must be between 0 and 23'}), 400
        tips = safety_predictor.get_safety_tips(ward_id, hour)
        return jsonify(tips)
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error retrieving safety tips: {str(e)}")
        return jsonify({'error': 'Failed to retrieve safety tips'}), 500

@app.route('/api/search')
def search_location():
    """API endpoint to search for a location and map it to the nearest ward."""
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
            
        # Use data processor to map the search query to ward coordinates
        ward_data = data_processor.map_search_query_to_ward(query)
        if not ward_data:
            return jsonify({'error': 'Location not found or not in Mumbai area'}), 404
            
        return jsonify(ward_data)
    except Exception as e:
        logger.error(f"Error during location search: {str(e)}")
        return jsonify({'error': 'Failed to search location'}), 500

# Create all database tables
with app.app_context():
    import models  # noqa: F401
    db.create_all()
