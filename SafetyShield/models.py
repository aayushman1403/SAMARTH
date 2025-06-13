from app import db
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    role = db.Column(db.String(20), default='user')  # e.g., 'admin', 'police', 'user'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Ward(db.Model):
    """Model to store Mumbai ward information."""
    id = db.Column(db.Integer, primary_key=True)
    ward_id = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    population = db.Column(db.Integer)
    area_sqkm = db.Column(db.Float)
    geo_json = db.Column(db.Text)  # Store GeoJSON representation if needed

class CrimeIncident(db.Model):
    """Model to store crime incident data."""
    id = db.Column(db.Integer, primary_key=True)
    incident_type = db.Column(db.String(100), nullable=False)
    ward_id = db.Column(db.String(10), db.ForeignKey('ward.ward_id'), nullable=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, nullable=False)
    severity = db.Column(db.Integer)  # Scale of 1-5 where 5 is most severe
    description = db.Column(db.Text)
    
    # Relationship with Ward
    ward = db.relationship('Ward', backref=db.backref('incidents', lazy=True))

class SafetyPrediction(db.Model):
    """Model to store safety predictions for wards by hour."""
    id = db.Column(db.Integer, primary_key=True)
    ward_id = db.Column(db.String(10), db.ForeignKey('ward.ward_id'), nullable=False)
    hour = db.Column(db.Integer, nullable=False)  # 0-23
    safety_level = db.Column(db.String(10), nullable=False)  # 'green', 'yellow', 'red'
    crime_probability = db.Column(db.Float)  # Probability score
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with Ward
    ward = db.relationship('Ward', backref=db.backref('predictions', lazy=True))
    
    # Composite unique constraint to ensure one prediction per ward per hour
    __table_args__ = (db.UniqueConstraint('ward_id', 'hour'),)
