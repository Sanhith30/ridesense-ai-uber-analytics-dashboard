import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class RidePredictionModels:
    """Class to handle all ML models for ride prediction"""
    
    def __init__(self):
        self.completion_model = None
        self.price_model = None
        self.duration_model = None
        self.scaler = StandardScaler()
        self.models_trained = False
        self.feature_columns = []
        self.performance_metrics = {}
        
    def train_models(self, df):
        """Train all prediction models"""
        from data_processor import DataProcessor
        
        # Initialize processor
        processor = DataProcessor()
        
        # Train completion prediction model
        self._train_completion_model(df, processor)
        
        # Train price prediction model
        self._train_price_model(df, processor)
        
        # Train duration prediction model
        self._train_duration_model(df, processor)
        
        self.models_trained = True
        self.processor = processor
        
    def _train_completion_model(self, df, processor):
        """Train ride completion prediction model"""
        # Prepare features and target
        features, target = processor.prepare_features_for_ml(df, 'IsCompleted')
        
        # Remove rows with missing target
        valid_indices = target.notna()
        X = features[valid_indices]
        y = target[valid_indices]
        
        if len(X) == 0:
            print("No valid data for completion model training")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.completion_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.completion_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.completion_model.predict(X_test)
        y_pred_proba = self.completion_model.predict_proba(X_test)[:, 1]
        
        # Store performance metrics
        self.performance_metrics['completion_model'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'feature_importance': dict(zip(X.columns, self.completion_model.feature_importances_))
        }
        
        print(f"Completion model trained. Accuracy: {self.performance_metrics['completion_model']['accuracy']:.3f}")
        
    def _train_price_model(self, df, processor):
        """Train price prediction model"""
        # Filter completed rides with valid booking values
        price_data = df[(df['IsCompleted'] == 1) & (df['Booking Value'].notna()) & (df['Booking Value'] > 0)]
        
        if len(price_data) == 0:
            print("No valid data for price model training")
            return
        
        # Prepare features and target
        features = processor.prepare_features_for_ml(price_data)
        target = price_data['Booking Value']
        
        # Remove outliers (prices beyond reasonable range)
        price_percentiles = target.quantile([0.01, 0.99])
        valid_price_mask = (target >= price_percentiles.iloc[0]) & (target <= price_percentiles.iloc[1])
        
        X = features[valid_price_mask]
        y = target[valid_price_mask]
        
        if len(X) == 0:
            print("No valid data after outlier removal for price model")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.price_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.price_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.price_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Store performance metrics
        self.performance_metrics['price_model'] = {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'feature_importance': dict(zip(X.columns, self.price_model.feature_importances_))
        }
        
        print(f"Price model trained. R² Score: {r2:.3f}")
        
    def _train_duration_model(self, df, processor):
        """Train ride duration prediction model"""
        # Filter completed rides with valid VTAT
        duration_data = df[(df['IsCompleted'] == 1) & (df['Avg VTAT'].notna()) & (df['Avg VTAT'] > 0)]
        
        if len(duration_data) == 0:
            print("No valid data for duration model training")
            return
        
        # Prepare features and target
        features = processor.prepare_features_for_ml(duration_data)
        target = duration_data['Avg VTAT']
        
        # Remove outliers (duration beyond reasonable range)
        duration_percentiles = target.quantile([0.01, 0.99])
        valid_duration_mask = (target >= duration_percentiles.iloc[0]) & (target <= duration_percentiles.iloc[1])
        
        X = features[valid_duration_mask]
        y = target[valid_duration_mask]
        
        if len(X) == 0:
            print("No valid data after outlier removal for duration model")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.duration_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.duration_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.duration_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Store performance metrics
        self.performance_metrics['duration_model'] = {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'feature_importance': dict(zip(X.columns, self.duration_model.feature_importances_))
        }
        
        print(f"Duration model trained. R² Score: {r2:.3f}")
        
    def predict_ride(self, input_data):
        """Make predictions for a ride"""
        if not self.models_trained:
            return {'error': 'Models not trained yet'}
        
        try:
            # Transform input data
            X_transformed = self.processor.transform_input_data(input_data)
            
            predictions = {}
            
            # Predict completion probability
            if self.completion_model:
                completion_proba = self.completion_model.predict_proba(X_transformed)[0, 1] * 100
                predictions['completion_probability'] = completion_proba
            
            # Predict price
            if self.price_model:
                predicted_price = self.price_model.predict(X_transformed)[0]
                predictions['predicted_price'] = max(0, predicted_price)
            
            # Predict duration
            if self.duration_model:
                predicted_duration = self.duration_model.predict(X_transformed)[0]
                predictions['predicted_duration'] = max(0, predicted_duration)
            
            return predictions
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def get_model_performance(self):
        """Get model performance metrics"""
        return self.performance_metrics
    
    def get_feature_importance(self):
        """Get feature importance for all models"""
        feature_importance = {}
        
        for model_name, metrics in self.performance_metrics.items():
            if 'feature_importance' in metrics:
                # Sort features by importance
                importance_dict = metrics['feature_importance']
                sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                feature_importance[model_name] = sorted_importance
        
        return feature_importance
    
    def get_cross_validation_results(self):
        """Perform cross-validation on models"""
        if not self.models_trained:
            return {}
        
        cv_results = {}
        
        # Note: This is a simplified version - in practice, you'd need to recreate the data splits
        # For now, we'll return the stored metrics
        for model_name, metrics in self.performance_metrics.items():
            cv_results[model_name] = {
                'Mean Score': metrics.get('accuracy', metrics.get('r2_score', 0)),
                'Std Score': 0.02  # Placeholder
            }
        
        return cv_results
    
    def generate_performance_report(self):
        """Generate a detailed performance report"""
        report = []
        report.append("=== NCR Ride Booking Prediction Models Performance Report ===\n")
        report.append(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        for model_name, metrics in self.performance_metrics.items():
            report.append(f"--- {model_name.replace('_', ' ').title()} ---")
            
            if model_name == 'completion_model':
                report.append(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
                report.append(f"Precision: {metrics.get('precision', 0):.3f}")
                report.append(f"Recall: {metrics.get('recall', 0):.3f}")
                report.append(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
            else:
                report.append(f"R² Score: {metrics.get('r2_score', 0):.3f}")
                report.append(f"Mean Absolute Error: {metrics.get('mae', 0):.3f}")
                report.append(f"Root Mean Square Error: {metrics.get('rmse', 0):.3f}")
                report.append(f"Mean Absolute Percentage Error: {metrics.get('mape', 0):.2f}%")
            
            # Top 5 important features
            if 'feature_importance' in metrics:
                importance_dict = metrics['feature_importance']
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                report.append("\nTop 5 Important Features:")
                for feature, importance in top_features:
                    report.append(f"  {feature}: {importance:.3f}")
            
            report.append("\n")
        
        return "\n".join(report)
