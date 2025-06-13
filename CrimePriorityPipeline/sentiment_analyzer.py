import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    """Class to perform sentiment analysis on text data"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a given text and return sentiment scores
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Dictionary containing sentiment scores (compound, positive, negative, neutral)
        """
        if not isinstance(text, str) or not text:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
        
        sentiment_scores = self.analyzer.polarity_scores(text)
        return sentiment_scores
    
    def add_sentiment_features(self, df, text_column='Description'):
        """
        Add sentiment features to the dataframe
        
        Args:
            df (pandas.DataFrame): The dataframe to add sentiment features to
            text_column (str): The column containing the text to analyze
            
        Returns:
            pandas.DataFrame: Dataframe with added sentiment features
        """
        # Make a copy to avoid modifying the original dataframe
        result_df = df.copy()
        
        # Check if the text column exists
        if text_column not in result_df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataframe")
        
        # Apply sentiment analysis to each row and create a new dataframe with sentiment scores
        sentiment_scores = result_df[text_column].apply(
            lambda text: self.analyze_sentiment(text)
        )
        
        # Extract sentiment scores into separate columns
        result_df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        result_df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['pos'])
        result_df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['neg'])
        result_df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['neu'])
        
        # Add a derived sentiment category column based on compound score
        result_df['sentiment_category'] = result_df['sentiment_compound'].apply(
            lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
        )
        
        return result_df
    
    def analyze_crime_sentiment_patterns(self, df):
        """
        Analyze sentiment patterns across different crime types
        
        Args:
            df (pandas.DataFrame): The dataframe with sentiment features
            
        Returns:
            pandas.DataFrame: Summary of sentiment by crime type
        """
        # Ensure necessary columns exist
        required_columns = ['CrimeType', 'sentiment_compound', 'sentiment_positive', 
                            'sentiment_negative', 'sentiment_neutral']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Required sentiment columns not found in the dataframe")
        
        # Group by crime type and calculate average sentiment scores
        sentiment_by_crime = df.groupby('CrimeType').agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'PriorityLabel': 'mean'  # Average priority to see correlation
        }).reset_index()
        
        # Sort by average compound sentiment score
        sentiment_by_crime = sentiment_by_crime.sort_values('sentiment_compound', ascending=True)
        
        return sentiment_by_crime

if __name__ == "__main__":
    # Test the sentiment analyzer
    from data_preprocessor import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('crime_data_synthetic.xlsx')
    
    if df is not None:
        processed_df = preprocessor.preprocess_data(df)
        
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_df = sentiment_analyzer.add_sentiment_features(processed_df)
        
        print(f"Sentiment analysis completed. Dataframe shape: {sentiment_df.shape}")
        
        # Display sentiment summary by crime type
        sentiment_summary = sentiment_analyzer.analyze_crime_sentiment_patterns(sentiment_df)
        print("\nSentiment Summary by Crime Type:")
        print(sentiment_summary)
