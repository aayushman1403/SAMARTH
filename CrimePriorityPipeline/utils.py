import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import os
import re
from datetime import datetime

def create_sample_case(crime_type=None, location=None, description=None, filing_date=None):
    """
    Create a sample case dictionary for prediction
    
    Args:
        crime_type (str): Type of crime
        location (str): Location of crime
        description (str): Description of crime
        filing_date (str): Filing date in format YYYY-MM-DD
        
    Returns:
        dict: Dictionary containing case information
    """
    # Default values if not provided
    if crime_type is None:
        crime_type = "Theft"
    
    if location is None:
        location = "Downtown"
    
    if description is None:
        description = "Victim reported a wallet stolen from a restaurant. No witnesses present."
    
    if filing_date is None:
        filing_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create case dictionary
    case = {
        'CaseID': f"NEW-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'FilingDate': filing_date,
        'CrimeType': crime_type,
        'Location': location,
        'Description': description
    }
    
    return case

def prepare_case_for_prediction(case, preprocessor, sentiment_analyzer, feature_engineer):
    """
    Prepare a case for prediction by applying preprocessing and feature engineering
    
    Args:
        case (dict): Case dictionary
        preprocessor: DataPreprocessor instance
        sentiment_analyzer: SentimentAnalyzer instance
        feature_engineer: FeatureEngineer instance
        
    Returns:
        pandas.DataFrame: Prepared feature matrix for the case
    """
    # Convert case to DataFrame
    case_df = pd.DataFrame([case])
    
    # Ensure FilingDate is datetime
    case_df['FilingDate'] = pd.to_datetime(case_df['FilingDate'])
    
    # Apply preprocessing
    case_df['CleanedDescription'] = case_df['Description'].apply(preprocessor.clean_text)
    
    # Apply sentiment analysis
    case_df = sentiment_analyzer.add_sentiment_features(case_df)
    
    # Apply feature engineering (with train=False to use pre-fitted transformers)
    case_df = feature_engineer.engineer_features(case_df, train=False)
    
    # Prepare for model input
    X, _ = feature_engineer.prepare_model_input(case_df)
    
    return X

def score_case(case, model_pipeline):
    """
    Score a case using the trained model pipeline
    
    Args:
        case (dict): Case dictionary
        model_pipeline: Dictionary containing preprocessor, sentiment_analyzer, feature_engineer, and model_builder
        
    Returns:
        tuple: Prediction (0/1) and probability
    """
    # Extract components from pipeline
    preprocessor = model_pipeline['preprocessor']
    sentiment_analyzer = model_pipeline['sentiment_analyzer']
    feature_engineer = model_pipeline['feature_engineer']
    model_builder = model_pipeline['model_builder']
    
    # Prepare case for prediction
    X = prepare_case_for_prediction(case, preprocessor, sentiment_analyzer, feature_engineer)
    
    # Make prediction
    prediction, probability = model_builder.predict_priority(X)
    
    return prediction[0], probability[0]

def plot_roc_curve(y_true, y_scores, figsize=(10, 6)):
    """
    Plot ROC curve for model evaluation
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    return plt.gcf()

def plot_precision_recall_curve(y_true, y_scores, figsize=(10, 6)):
    """
    Plot Precision-Recall curve for model evaluation
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6)):
    """
    Plot confusion matrix for model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    return plt.gcf()

def plot_feature_importances(feature_importances, top_n=10, figsize=(12, 8)):
    """
    Plot top feature importances
    
    Args:
        feature_importances: DataFrame with feature importances
        top_n: Number of top features to plot
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Get top N features
    top_features = feature_importances.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    return plt.gcf()

def plot_sentiment_distribution(df, figsize=(10, 6)):
    """
    Plot sentiment distribution by priority
    
    Args:
        df: DataFrame with sentiment features and priority labels
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=figsize)
    
    # Create separate data for high and low priority cases
    high_priority = df[df['PriorityLabel'] == 1]['sentiment_compound']
    low_priority = df[df['PriorityLabel'] == 0]['sentiment_compound']
    
    # Plot distributions
    sns.kdeplot(high_priority, label='High Priority', color='red')
    sns.kdeplot(low_priority, label='Low Priority', color='blue')
    
    plt.title('Sentiment Distribution by Priority')
    plt.xlabel('Compound Sentiment Score')
    plt.ylabel('Density')
    plt.legend()
    
    return plt.gcf()

def plot_crime_type_distribution(df, figsize=(12, 8)):
    """
    Plot crime type distribution
    
    Args:
        df: DataFrame with crime types
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=figsize)
    
    # Count crime types
    crime_counts = df['CrimeType'].value_counts()
    
    # Create bar plot
    sns.barplot(x=crime_counts.index, y=crime_counts.values)
    plt.title('Crime Type Distribution')
    plt.xlabel('Crime Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()
