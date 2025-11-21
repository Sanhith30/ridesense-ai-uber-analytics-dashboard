import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from ml_models import RidePredictionModels
from visualizations import RideAnalyticsDashboard
from utils import load_data, format_currency, format_duration

# Set page configuration
st.set_page_config(
    page_title="NCR Ride Booking Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Title and header
st.title("🚗 NCR Ride Booking Prediction Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard", "Predict Ride", "Data Analytics", "Model Performance"]
)

# Load data
@st.cache_data
def load_and_process_data():
    """Load and process the ride booking data"""
    try:
        # Try to load the CSV file
        data_path = "attached_assets/ncr_ride_bookings_1755850616781.csv"
        df = pd.read_csv(data_path)
        
        # Initialize data processor
        processor = DataProcessor()
        processed_df = processor.process_data(df)
        
        return processed_df, processor
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Initialize data and processor
if not st.session_state.data_loaded:
    with st.spinner("Loading and processing data..."):
        df, processor = load_and_process_data()
        if df is not None:
            st.session_state.df = df
            st.session_state.processor = processor
            st.session_state.data_loaded = True
            
            # Initialize and train models
            with st.spinner("Training ML models..."):
                models = RidePredictionModels()
                models.train_models(df)
                st.session_state.models = models
                st.session_state.models_trained = True
        else:
            st.stop()

# Get data from session state
df = st.session_state.df
processor = st.session_state.processor
models = st.session_state.models if st.session_state.models_trained else None

# Dashboard Page
if page == "Dashboard":
    st.header("📊 Ride Booking Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_bookings = len(df)
        st.metric("Total Bookings", f"{total_bookings:,}")
    
    with col2:
        completed_rides = len(df[df['Booking Status'] == 'Completed'])
        completion_rate = (completed_rides / total_bookings) * 100
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    with col3:
        avg_booking_value = df[df['Booking Value'].notna()]['Booking Value'].mean()
        st.metric("Avg Booking Value", format_currency(avg_booking_value))
    
    with col4:
        avg_distance = df[df['Ride Distance'].notna()]['Ride Distance'].mean()
        st.metric("Avg Distance", f"{avg_distance:.1f} km")
    
    st.markdown("---")
    
    # Charts
    dashboard = RideAnalyticsDashboard()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Booking Status Distribution")
        status_fig = dashboard.create_booking_status_chart(df)
        st.plotly_chart(status_fig, use_container_width=True)
    
    with col2:
        st.subheader("Vehicle Type Distribution")
        vehicle_fig = dashboard.create_vehicle_type_chart(df)
        st.plotly_chart(vehicle_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Booking Pattern")
        hourly_fig = dashboard.create_hourly_pattern_chart(df)
        st.plotly_chart(hourly_fig, use_container_width=True)
    
    with col2:
        st.subheader("Payment Method Distribution")
        payment_fig = dashboard.create_payment_method_chart(df)
        st.plotly_chart(payment_fig, use_container_width=True)

# Predict Ride Page
elif page == "Predict Ride":
    st.header("🔮 Ride Prediction")
    
    if not st.session_state.models_trained:
        st.error("Models are not trained yet. Please wait for the training to complete.")
        st.stop()
    
    st.markdown("Enter ride details to get predictions for completion probability, price, and duration.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ride Details")
        
        # Vehicle type selection
        vehicle_types = df['Vehicle Type'].unique()
        vehicle_type = st.selectbox("Vehicle Type", vehicle_types)
        
        # Location selection
        pickup_locations = df['Pickup Location'].unique()
        pickup_location = st.selectbox("Pickup Location", pickup_locations)
        
        drop_locations = df['Drop Location'].unique()
        drop_location = st.selectbox("Drop Location", drop_locations)
        
        # Date and time
        booking_date = st.date_input("Booking Date", datetime.now().date())
        booking_time = st.time_input("Booking Time", datetime.now().time())
        
        # Payment method
        payment_methods = df['Payment Method'].dropna().unique()
        payment_method = st.selectbox("Payment Method", payment_methods)
        
    with col2:
        st.subheader("Predictions")
        
        if st.button("Get Predictions", type="primary"):
            # Prepare input data
            input_data = {
                'Vehicle Type': vehicle_type,
                'Pickup Location': pickup_location,
                'Drop Location': drop_location,
                'Date': booking_date,
                'Time': booking_time,
                'Payment Method': payment_method,
                'Hour': booking_time.hour,
                'DayOfWeek': booking_date.weekday(),
                'Month': booking_date.month
            }
            
            # Get predictions
            predictions = models.predict_ride(input_data)
            
            st.markdown("### 📈 Prediction Results")
            
            # Completion probability
            completion_prob = predictions.get('completion_probability', 0)
            st.metric(
                "Completion Probability",
                f"{completion_prob:.1f}%",
                delta=f"{completion_prob - 75:.1f}%" if completion_prob > 75 else f"{completion_prob - 75:.1f}%"
            )
            
            # Price prediction
            predicted_price = predictions.get('predicted_price', 0)
            st.metric(
                "Predicted Price",
                format_currency(predicted_price),
                help="Estimated fare for this ride"
            )
            
            # Duration prediction
            predicted_duration = predictions.get('predicted_duration', 0)
            st.metric(
                "Predicted Duration",
                format_duration(predicted_duration),
                help="Estimated ride duration in minutes"
            )
            
            # Risk factors
            st.markdown("### ⚠️ Risk Assessment")
            risk_level = "Low" if completion_prob > 80 else "Medium" if completion_prob > 60 else "High"
            risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
            
            st.markdown(f"**Cancellation Risk:** :{risk_color}[{risk_level}]")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if completion_prob < 70:
                st.warning("⚠️ High cancellation risk. Consider offering incentives or choosing alternative timing.")
            elif completion_prob < 85:
                st.info("ℹ️ Moderate risk. Monitor closely for any driver/customer issues.")
            else:
                st.success("✅ High probability of successful completion!")

# Data Analytics Page
elif page == "Data Analytics":
    st.header("📈 Data Analytics")
    
    # Analytics options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Temporal Analysis", "Location Analysis", "Vehicle Analysis", "Cancellation Analysis"]
    )
    
    dashboard = RideAnalyticsDashboard()
    
    if analysis_type == "Temporal Analysis":
        st.subheader("📅 Temporal Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly trend
            monthly_fig = dashboard.create_monthly_trend_chart(df)
            st.plotly_chart(monthly_fig, use_container_width=True)
        
        with col2:
            # Daily pattern
            daily_fig = dashboard.create_daily_pattern_chart(df)
            st.plotly_chart(daily_fig, use_container_width=True)
        
        # Heatmap
        st.subheader("Booking Heatmap (Hour vs Day of Week)")
        heatmap_fig = dashboard.create_booking_heatmap(df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    elif analysis_type == "Location Analysis":
        st.subheader("📍 Location Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Pickup Locations**")
            top_pickups = df['Pickup Location'].value_counts().head(10)
            pickup_fig = px.bar(
                x=top_pickups.values,
                y=top_pickups.index,
                orientation='h',
                title="Most Popular Pickup Locations"
            )
            pickup_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(pickup_fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top Drop Locations**")
            top_drops = df['Drop Location'].value_counts().head(10)
            drop_fig = px.bar(
                x=top_drops.values,
                y=top_drops.index,
                orientation='h',
                title="Most Popular Drop Locations"
            )
            drop_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(drop_fig, use_container_width=True)
        
        # Route analysis
        st.subheader("Popular Routes")
        df['Route'] = df['Pickup Location'] + ' → ' + df['Drop Location']
        top_routes = df['Route'].value_counts().head(20)
        route_fig = px.bar(
            x=top_routes.values,
            y=top_routes.index,
            orientation='h',
            title="Top 20 Most Popular Routes"
        )
        route_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(route_fig, use_container_width=True)
    
    elif analysis_type == "Vehicle Analysis":
        st.subheader("🚗 Vehicle Type Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average booking value by vehicle type
            avg_value_by_vehicle = df.groupby('Vehicle Type')['Booking Value'].mean().sort_values(ascending=False)
            value_fig = px.bar(
                x=avg_value_by_vehicle.index,
                y=avg_value_by_vehicle.values,
                title="Average Booking Value by Vehicle Type"
            )
            st.plotly_chart(value_fig, use_container_width=True)
        
        with col2:
            # Completion rate by vehicle type
            vehicle_completion = df.groupby('Vehicle Type')['Booking Status'].apply(
                lambda x: (x == 'Completed').mean() * 100
            ).sort_values(ascending=False)
            completion_fig = px.bar(
                x=vehicle_completion.index,
                y=vehicle_completion.values,
                title="Completion Rate by Vehicle Type (%)"
            )
            st.plotly_chart(completion_fig, use_container_width=True)
        
        # Distance vs Price analysis
        st.subheader("Distance vs Price Analysis")
        completed_rides = df[df['Booking Status'] == 'Completed']
        if not completed_rides.empty:
            scatter_fig = px.scatter(
                completed_rides,
                x='Ride Distance',
                y='Booking Value',
                color='Vehicle Type',
                title="Ride Distance vs Booking Value",
                trendline="ols"
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
    
    elif analysis_type == "Cancellation Analysis":
        st.subheader("❌ Cancellation Analysis")
        
        # Cancellation reasons
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer cancellation reasons
            customer_cancellations = df[df['Cancelled Rides by Customer'] == 1]['Reason for cancelling by Customer'].value_counts()
            if not customer_cancellations.empty:
                customer_fig = px.pie(
                    values=customer_cancellations.values,
                    names=customer_cancellations.index,
                    title="Customer Cancellation Reasons"
                )
                st.plotly_chart(customer_fig, use_container_width=True)
        
        with col2:
            # Driver cancellation reasons
            driver_cancellations = df[df['Cancelled Rides by Driver'] == 1]['Driver Cancellation Reason'].value_counts()
            if not driver_cancellations.empty:
                driver_fig = px.pie(
                    values=driver_cancellations.values,
                    names=driver_cancellations.index,
                    title="Driver Cancellation Reasons"
                )
                st.plotly_chart(driver_fig, use_container_width=True)
        
        # Cancellation rate by hour
        df_with_hour = df.copy()
        df_with_hour['Hour'] = pd.to_datetime(df_with_hour['Time'], format='%H:%M:%S').dt.hour
        hourly_cancellation = df_with_hour.groupby('Hour').apply(
            lambda x: ((x['Booking Status'] == 'Cancelled by Customer') | 
                      (x['Booking Status'] == 'Cancelled by Driver')).mean() * 100
        )
        
        cancellation_hourly_fig = px.line(
            x=hourly_cancellation.index,
            y=hourly_cancellation.values,
            title="Cancellation Rate by Hour of Day (%)",
            markers=True
        )
        st.plotly_chart(cancellation_hourly_fig, use_container_width=True)

# Model Performance Page
elif page == "Model Performance":
    st.header("🎯 Model Performance")
    
    if not st.session_state.models_trained:
        st.error("Models are not trained yet. Please wait for the training to complete.")
        st.stop()
    
    # Model performance metrics
    performance_metrics = models.get_model_performance()
    
    st.subheader("📊 Model Accuracy Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Completion Prediction Model**")
        completion_metrics = performance_metrics.get('completion_model', {})
        st.metric("Accuracy", f"{completion_metrics.get('accuracy', 0):.3f}")
        st.metric("Precision", f"{completion_metrics.get('precision', 0):.3f}")
        st.metric("Recall", f"{completion_metrics.get('recall', 0):.3f}")
        st.metric("F1 Score", f"{completion_metrics.get('f1_score', 0):.3f}")
    
    with col2:
        st.markdown("**Price Prediction Model**")
        price_metrics = performance_metrics.get('price_model', {})
        st.metric("R² Score", f"{price_metrics.get('r2_score', 0):.3f}")
        st.metric("MAE", f"{price_metrics.get('mae', 0):.2f}")
        st.metric("RMSE", f"{price_metrics.get('rmse', 0):.2f}")
        st.metric("MAPE", f"{price_metrics.get('mape', 0):.2f}%")
    
    with col3:
        st.markdown("**Duration Prediction Model**")
        duration_metrics = performance_metrics.get('duration_model', {})
        st.metric("R² Score", f"{duration_metrics.get('r2_score', 0):.3f}")
        st.metric("MAE", f"{duration_metrics.get('mae', 0):.2f} min")
        st.metric("RMSE", f"{duration_metrics.get('rmse', 0):.2f} min")
        st.metric("MAPE", f"{duration_metrics.get('mape', 0):.2f}%")
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    
    feature_importance = models.get_feature_importance()
    
    for model_name, importance_data in feature_importance.items():
        if importance_data:
            st.markdown(f"**{model_name.replace('_', ' ').title()}**")
            
            importance_df = pd.DataFrame(
                list(importance_data.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df.tail(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 10 Important Features - {model_name.replace('_', ' ').title()}"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.subheader("📈 Model Comparison")
    
    # Cross-validation results
    cv_results = models.get_cross_validation_results()
    if cv_results:
        cv_df = pd.DataFrame(cv_results).T
        st.dataframe(cv_df, use_container_width=True)
    
    # Download model performance report
    if st.button("📄 Download Performance Report"):
        report = models.generate_performance_report()
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"model_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("**NCR Ride Booking Prediction Dashboard** | Powered by Machine Learning")
