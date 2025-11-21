# NCR Ride Booking Prediction Dashboard

## Overview

This is a Streamlit-based machine learning dashboard for analyzing and predicting ride booking patterns in the National Capital Region (NCR). The application provides comprehensive analytics on ride bookings including status distribution, vehicle types, hourly patterns, and predictive capabilities for ride completion, pricing, and duration. The system processes CSV data containing ride booking information and uses various machine learning models to generate insights and predictions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Web-based dashboard using Streamlit for interactive data visualization and user interface
- **Multi-page Navigation**: Sidebar-based navigation system with pages for Dashboard, Predict Ride, Data Analytics, and Model Performance
- **Session State Management**: Persistent state management for data loading and model training status
- **Caching Strategy**: Uses `@st.cache_data` decorator for efficient data loading and processing

### Backend Architecture
- **Modular Design**: Code organized into separate modules for data processing, ML models, visualizations, and utilities
- **Data Processing Pipeline**: `DataProcessor` class handles data cleaning, feature engineering, and preprocessing
- **Machine Learning Engine**: `RidePredictionModels` class manages multiple ML models for different prediction tasks
- **Visualization Engine**: `RideAnalyticsDashboard` class creates interactive charts and graphs

### Data Processing Layer
- **Feature Engineering**: Datetime processing, derived feature creation, and categorical variable handling
- **Data Cleaning**: Automatic removal of quotes from column names and values, missing value imputation
- **ML-Ready Pipeline**: Preparation of features for machine learning with proper encoding and scaling

### Machine Learning Models
- **Multi-target Prediction**: Separate models for ride completion, price prediction, and duration estimation
- **Ensemble Methods**: Uses RandomForest and GradientBoosting algorithms for robust predictions
- **Model Evaluation**: Comprehensive performance metrics including accuracy, precision, recall, F1-score, R2, MAE, and MSE
- **Cross-validation**: Built-in cross-validation for model reliability assessment

### Data Storage
- **CSV File Processing**: Handles CSV data input from the `attached_assets` directory
- **In-memory Processing**: All data processing and model training performed in memory
- **Model Persistence**: Capability to save and load trained models using joblib

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework for the dashboard interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **plotly**: Interactive visualization library (express and graph_objects)
- **scikit-learn**: Machine learning algorithms and utilities
- **joblib**: Model serialization and persistence
- **datetime**: Date and time processing

### Data Requirements
- **CSV Data Source**: Expects NCR ride booking data in CSV format
- **Required Columns**: Date, Time, Booking Status, Vehicle Type, pricing, and duration information
- **Data Format**: Handles quoted CSV values and performs automatic cleaning

### Visualization Components
- **Plotly Charts**: Interactive pie charts, bar charts, line plots, and subplots
- **Color Schemes**: Uses Plotly's qualitative color palettes for consistent styling
- **Responsive Design**: Charts adapt to different screen sizes and layouts