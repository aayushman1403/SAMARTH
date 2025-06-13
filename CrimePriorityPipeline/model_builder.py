import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

class ModelBuilder:
    """Class to build and evaluate the priority prediction model"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_importances = None
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets
        
        Args:
            X (pandas.DataFrame): Feature matrix
            y (pandas.Series): Target variable
            test_size (float): Proportion of the dataset to include in the test split
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, X_train, y_train, tune_hyperparameters=True):
        """
        Build and train the classification model
        
        Args:
            X_train (pandas.DataFrame): Training feature matrix
            y_train (pandas.Series): Training target variable
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained model
        """
        if tune_hyperparameters:
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            }
            
            # Initialize base model
            base_model = RandomForestClassifier(random_state=self.random_state)
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            print("Performing hyperparameter tuning...")
            grid_search.fit(X_train, y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use default hyperparameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=self.random_state
            )
            
            print("Training model with default hyperparameters...")
            self.model.fit(X_train, y_train)
        
        # Get feature importances
        self.feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Args:
            X_test (pandas.DataFrame): Testing feature matrix
            y_test (pandas.Series): Testing target variable
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call build_model first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        # Print evaluation results
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Print classification report
        report = classification_report(y_test, y_pred)
        print(f"\nClassification Report:\n{report}")
        
        # Return metrics as a dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return metrics
    
    def get_feature_importances(self):
        """
        Get feature importances from the trained model
        
        Returns:
            pandas.DataFrame: DataFrame with feature importances
        """
        if self.feature_importances is None:
            raise ValueError("Feature importances not available. Train the model first.")
        
        return self.feature_importances
    
    def save_model(self, filename='crime_priority_model.joblib'):
        """
        Save the trained model to a file
        
        Args:
            filename (str): Path to save the model
            
        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call build_model first.")
        
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
        
        return filename
    
    def load_model(self, filename='crime_priority_model.joblib'):
        """
        Load a trained model from a file
        
        Args:
            filename (str): Path to the saved model
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: Loaded model
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")
        
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        
        return self.model
    
    def predict_priority(self, X):
        """
        Predict priority for new cases
        
        Args:
            X (pandas.DataFrame): Feature matrix for new cases
            
        Returns:
            tuple: Predictions and probability scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call build_model first.")
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities

if __name__ == "__main__":
    # Test the model builder
    from data_preprocessor import DataPreprocessor
    from sentiment_analyzer import SentimentAnalyzer
    from feature_engineering import FeatureEngineer
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('crime_data_synthetic.xlsx')
    
    if df is not None:
        processed_df = preprocessor.preprocess_data(df)
        
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_df = sentiment_analyzer.add_sentiment_features(processed_df)
        
        feature_engineer = FeatureEngineer()
        engineered_df = feature_engineer.engineer_features(sentiment_df)
        
        X, y = feature_engineer.prepare_model_input(engineered_df)
        
        model_builder = ModelBuilder()
        X_train, X_test, y_train, y_test = model_builder.split_data(X, y)
        
        # Train model (with default hyperparameters for quick testing)
        model = model_builder.build_model(X_train, y_train, tune_hyperparameters=False)
        
        # Evaluate model
        metrics = model_builder.evaluate_model(X_test, y_test)
        
        # Get feature importances
        feature_importances = model_builder.get_feature_importances()
        print("\nTop 10 Feature Importances:")
        print(feature_importances.head(10))
        
        # Save model
        model_builder.save_model()
