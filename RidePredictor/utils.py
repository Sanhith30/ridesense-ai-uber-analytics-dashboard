import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os

def load_data(file_path):
    """Load CSV data with error handling"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def format_currency(amount):
    """Format amount as Indian currency"""
    if pd.isna(amount) or amount == 0:
        return "₹0"
    
    # Convert to float if it's not already
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return "₹0"
    
    # Format with Indian number system (lakhs, crores)
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.1f}Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.1f}L"
    elif amount >= 1000:  # 1 thousand
        return f"₹{amount/1000:.1f}K"
    else:
        return f"₹{amount:.0f}"

def format_duration(minutes):
    """Format duration in minutes to human readable format"""
    if pd.isna(minutes) or minutes == 0:
        return "0 min"
    
    try:
        minutes = float(minutes)
    except (ValueError, TypeError):
        return "0 min"
    
    if minutes >= 60:
        hours = int(minutes // 60)
        remaining_minutes = int(minutes % 60)
        if remaining_minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {remaining_minutes}m"
    else:
        return f"{int(minutes)}m"

def format_distance(km):
    """Format distance in kilometers"""
    if pd.isna(km) or km == 0:
        return "0 km"
    
    try:
        km = float(km)
    except (ValueError, TypeError):
        return "0 km"
    
    if km < 1:
        return f"{int(km * 1000)}m"
    else:
        return f"{km:.1f}km"

def calculate_completion_rate(df, group_by_column=None):
    """Calculate completion rate overall or by group"""
    if group_by_column and group_by_column in df.columns:
        completion_rates = df.groupby(group_by_column)['Booking Status'].apply(
            lambda x: (x == 'Completed').mean() * 100
        )
        return completion_rates
    else:
        total_bookings = len(df)
        completed_bookings = len(df[df['Booking Status'] == 'Completed'])
        return (completed_bookings / total_bookings * 100) if total_bookings > 0 else 0

def calculate_cancellation_rate(df, group_by_column=None):
    """Calculate cancellation rate overall or by group"""
    if group_by_column and group_by_column in df.columns:
        cancellation_rates = df.groupby(group_by_column)['Booking Status'].apply(
            lambda x: (x.str.contains('Cancel', na=False)).mean() * 100
        )
        return cancellation_rates
    else:
        total_bookings = len(df)
        cancelled_bookings = len(df[df['Booking Status'].str.contains('Cancel', na=False)])
        return (cancelled_bookings / total_bookings * 100) if total_bookings > 0 else 0

def get_peak_hours(df):
    """Get peak booking hours"""
    df_copy = df.copy()
    df_copy['Hour'] = pd.to_datetime(df_copy['Time'], format='%H:%M:%S').dt.hour
    hourly_counts = df_copy['Hour'].value_counts().sort_index()
    
    # Get top 3 peak hours
    peak_hours = hourly_counts.nlargest(3)
    return peak_hours.index.tolist()

def get_popular_routes(df, top_n=10):
    """Get most popular routes (pickup to drop location pairs)"""
    df_copy = df.copy()
    df_copy['Route'] = df_copy['Pickup Location'] + ' → ' + df_copy['Drop Location']
    popular_routes = df_copy['Route'].value_counts().head(top_n)
    return popular_routes

def calculate_average_metrics(df):
    """Calculate various average metrics"""
    metrics = {}
    
    # Average booking value
    completed_rides = df[df['Booking Status'] == 'Completed']
    metrics['avg_booking_value'] = completed_rides['Booking Value'].mean()
    
    # Average ride distance
    metrics['avg_distance'] = completed_rides['Ride Distance'].mean()
    
    # Average VTAT (Vehicle Travel and Arrival Time)
    metrics['avg_vtat'] = completed_rides['Avg VTAT'].mean()
    
    # Average CTAT (Customer Travel and Arrival Time)
    metrics['avg_ctat'] = completed_rides['Avg CTAT'].mean()
    
    # Average ratings
    metrics['avg_driver_rating'] = completed_rides['Driver Ratings'].mean()
    metrics['avg_customer_rating'] = completed_rides['Customer Rating'].mean()
    
    return metrics

def get_data_quality_report(df):
    """Generate data quality report"""
    report = {}
    
    # Total records
    report['total_records'] = len(df)
    
    # Missing values per column
    report['missing_values'] = df.isnull().sum().to_dict()
    
    # Data types
    report['data_types'] = df.dtypes.to_dict()
    
    # Unique values count
    report['unique_values'] = df.nunique().to_dict()
    
    # Completion rate by status
    status_counts = df['Booking Status'].value_counts()
    report['booking_status_distribution'] = status_counts.to_dict()
    
    return report

def export_predictions_to_csv(predictions_data, filename=None):
    """Export predictions to CSV file"""
    if filename is None:
        filename = f"ride_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    df = pd.DataFrame(predictions_data)
    return df.to_csv(index=False)

def validate_input_data(input_data):
    """Validate input data for predictions"""
    required_fields = ['Vehicle Type', 'Pickup Location', 'Drop Location', 'Date', 'Time']
    
    missing_fields = []
    for field in required_fields:
        if field not in input_data or input_data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, "Input data is valid"

def create_location_mapping(df):
    """Create mapping for locations to handle high cardinality"""
    # Get top locations by frequency
    pickup_counts = df['Pickup Location'].value_counts()
    drop_counts = df['Drop Location'].value_counts()
    
    # Keep top 50 locations for each
    top_pickups = pickup_counts.head(50).index.tolist()
    top_drops = drop_counts.head(50).index.tolist()
    
    return {
        'top_pickup_locations': top_pickups,
        'top_drop_locations': top_drops
    }

def get_model_recommendations(completion_probability, predicted_price, predicted_duration):
    """Get recommendations based on model predictions"""
    recommendations = []
    
    # Completion probability recommendations
    if completion_probability < 60:
        recommendations.append("⚠️ Very high cancellation risk. Consider offering promotional pricing or scheduling for off-peak hours.")
    elif completion_probability < 75:
        recommendations.append("🔶 Moderate cancellation risk. Monitor for driver availability and customer communication.")
    else:
        recommendations.append("✅ High completion probability. Good conditions for this ride.")
    
    # Price recommendations
    if predicted_price > 500:
        recommendations.append("💰 High fare ride. Consider premium service features.")
    elif predicted_price < 100:
        recommendations.append("💡 Low fare ride. Ensure cost efficiency.")
    
    # Duration recommendations
    if predicted_duration > 60:
        recommendations.append("⏰ Long duration ride. Inform customer about expected travel time.")
    elif predicted_duration > 30:
        recommendations.append("🕒 Medium duration ride. Standard service expectations.")
    
    return recommendations

def generate_summary_stats(df):
    """Generate summary statistics for the dashboard"""
    stats = {
        'total_bookings': len(df),
        'completed_rides': len(df[df['Booking Status'] == 'Completed']),
        'cancelled_rides': len(df[df['Booking Status'].str.contains('Cancel', na=False)]),
        'incomplete_rides': len(df[df['Booking Status'] == 'Incomplete']),
        'no_driver_found': len(df[df['Booking Status'] == 'No Driver Found']),
        'unique_customers': df['Customer ID'].nunique(),
        'unique_locations': df['Pickup Location'].nunique() + df['Drop Location'].nunique(),
        'date_range': {
            'start': df['Date'].min(),
            'end': df['Date'].max()
        }
    }
    
    # Calculate rates
    if stats['total_bookings'] > 0:
        stats['completion_rate'] = (stats['completed_rides'] / stats['total_bookings']) * 100
        stats['cancellation_rate'] = (stats['cancelled_rides'] / stats['total_bookings']) * 100
    else:
        stats['completion_rate'] = 0
        stats['cancellation_rate'] = 0
    
    return stats
