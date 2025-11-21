import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Class to handle data preprocessing and feature engineering"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = []
        
    def process_data(self, df):
        """Main method to process the raw data"""
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Clean column names (remove quotes)
        processed_df.columns = processed_df.columns.str.strip('"')
        
        # Clean data values (remove quotes)
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                processed_df[col] = processed_df[col].astype(str).str.strip('"')
        
        # Handle datetime columns
        processed_df = self._process_datetime(processed_df)
        
        # Handle numeric columns
        processed_df = self._process_numeric_columns(processed_df)
        
        # Create derived features
        processed_df = self._create_derived_features(processed_df)
        
        # Handle categorical variables
        processed_df = self._process_categorical_variables(processed_df)
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        return processed_df
    
    def _process_datetime(self, df):
        """Process date and time columns"""
        # Combine Date and Time columns
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        
        # Extract datetime components
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['Month'] = df['DateTime'].dt.month
        df['DayOfMonth'] = df['DateTime'].dt.day
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Create time periods
        df['TimePeriod'] = pd.cut(
            df['Hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        )
        
        return df
    
    def _process_numeric_columns(self, df):
        """Process numeric columns"""
        numeric_columns = [
            'Avg VTAT', 'Avg CTAT', 'Cancelled Rides by Customer',
            'Cancelled Rides by Driver', 'Incomplete Rides',
            'Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                # Replace 'null' strings with NaN
                df[col] = df[col].replace('null', np.nan)
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _create_derived_features(self, df):
        """Create derived features"""
        # Price per km
        df['PricePerKm'] = np.where(
            (df['Ride Distance'] > 0) & (df['Booking Value'].notna()),
            df['Booking Value'] / df['Ride Distance'],
            np.nan
        )
        
        # Speed (km/hour) - assuming VTAT is in minutes
        df['AverageSpeed'] = np.where(
            (df['Avg VTAT'] > 0) & (df['Ride Distance'] > 0),
            (df['Ride Distance'] / df['Avg VTAT']) * 60,
            np.nan
        )
        
        # Route popularity score (frequency of pickup-drop combination)
        route_counts = df.groupby(['Pickup Location', 'Drop Location']).size()
        df['RoutePopularity'] = df.apply(
            lambda row: route_counts.get((row['Pickup Location'], row['Drop Location']), 0),
            axis=1
        )
        
        # Customer and driver ratings difference
        df['RatingDifference'] = np.where(
            df['Driver Ratings'].notna() & df['Customer Rating'].notna(),
            df['Customer Rating'] - df['Driver Ratings'],
            np.nan
        )
        
        # Create binary target variables
        df['IsCompleted'] = (df['Booking Status'] == 'Completed').astype(int)
        df['IsCancelledByCustomer'] = (df['Booking Status'] == 'Cancelled by Customer').astype(int)
        df['IsCancelledByDriver'] = (df['Booking Status'] == 'Cancelled by Driver').astype(int)
        df['IsIncomplete'] = (df['Booking Status'] == 'Incomplete').astype(int)
        
        return df
    
    def _process_categorical_variables(self, df):
        """Process categorical variables"""
        # Define categorical columns
        categorical_columns = [
            'Vehicle Type', 'Pickup Location', 'Drop Location',
            'Payment Method', 'TimePeriod'
        ]
        
        # Handle high-cardinality categorical variables
        for col in ['Pickup Location', 'Drop Location']:
            if col in df.columns:
                # Keep only top locations, group others as 'Other'
                top_locations = df[col].value_counts().head(50).index
                df[f'{col}_Grouped'] = df[col].apply(
                    lambda x: x if x in top_locations else 'Other'
                )
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values"""
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        return df
    
    def prepare_features_for_ml(self, df, target_column=None):
        """Prepare features for machine learning"""
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        
        # Define feature columns
        numeric_features = [
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
            'Avg VTAT', 'Avg CTAT', 'Ride Distance',
            'Driver Ratings', 'Customer Rating', 'RoutePopularity'
        ]
        
        categorical_features = [
            'Vehicle Type', 'Pickup Location_Grouped', 'Drop Location_Grouped',
            'Payment Method', 'TimePeriod'
        ]
        
        # Select available features
        available_numeric = [col for col in numeric_features if col in df.columns]
        available_categorical = [col for col in categorical_features if col in df.columns]
        
        feature_df = df[available_numeric + available_categorical].copy()
        
        # Encode categorical variables
        for col in available_categorical:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                feature_df[col] = self.label_encoders[col].fit_transform(feature_df[col].astype(str))
            else:
                # Transform using existing encoder
                feature_df[col] = self.label_encoders[col].transform(feature_df[col].astype(str))
        
        self.feature_columns = feature_df.columns.tolist()
        
        if target_column and target_column in df.columns:
            return feature_df, df[target_column]
        else:
            return feature_df
    
    def transform_input_data(self, input_data):
        """Transform input data for prediction"""
        # Create a dataframe from input
        input_df = pd.DataFrame([input_data])
        
        # Apply same transformations
        input_df['IsWeekend'] = (input_df['DayOfWeek'] >= 5).astype(int)
        input_df['TimePeriod'] = pd.cut(
            input_df['Hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        )
        
        # Group locations (simplified - in practice, you'd use the same grouping logic)
        input_df['Pickup Location_Grouped'] = input_data.get('Pickup Location', 'Other')
        input_df['Drop Location_Grouped'] = input_data.get('Drop Location', 'Other')
        
        # Set default values for missing features
        default_values = {
            'Avg VTAT': 10.0,
            'Avg CTAT': 25.0,
            'Ride Distance': 15.0,
            'Driver Ratings': 4.2,
            'Customer Rating': 4.3,
            'RoutePopularity': 1
        }
        
        for col, default_val in default_values.items():
            if col not in input_df.columns:
                input_df[col] = default_val
        
        # Encode categorical variables
        categorical_features = [
            'Vehicle Type', 'Pickup Location_Grouped', 'Drop Location_Grouped',
            'Payment Method', 'TimePeriod'
        ]
        
        for col in categorical_features:
            if col in input_df.columns and col in self.label_encoders:
                try:
                    input_df[col] = self.label_encoders[col].transform(input_df[col].astype(str))
                except ValueError:
                    # Handle unknown categories
                    input_df[col] = 0
        
        # Select only the features used in training
        feature_cols = [col for col in self.feature_columns if col in input_df.columns]
        return input_df[feature_cols]
