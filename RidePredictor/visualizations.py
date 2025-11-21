import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

class RideAnalyticsDashboard:
    """Class to create various visualizations for ride analytics"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def create_booking_status_chart(self, df):
        """Create booking status distribution pie chart"""
        status_counts = df['Booking Status'].value_counts()
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Booking Status Distribution",
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True)
        
        return fig
    
    def create_vehicle_type_chart(self, df):
        """Create vehicle type distribution chart"""
        vehicle_counts = df['Vehicle Type'].value_counts()
        
        fig = px.bar(
            x=vehicle_counts.index,
            y=vehicle_counts.values,
            title="Vehicle Type Distribution",
            color=vehicle_counts.index,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title="Vehicle Type",
            yaxis_title="Number of Bookings",
            showlegend=False
        )
        
        return fig
    
    def create_hourly_pattern_chart(self, df):
        """Create hourly booking pattern chart"""
        # Extract hour from time
        df_copy = df.copy()
        df_copy['Hour'] = pd.to_datetime(df_copy['Time'], format='%H:%M:%S').dt.hour
        
        hourly_counts = df_copy.groupby('Hour').size()
        
        fig = px.line(
            x=hourly_counts.index,
            y=hourly_counts.values,
            title="Hourly Booking Pattern",
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Number of Bookings"
        )
        
        fig.update_traces(line_color='#1f77b4', marker_size=8)
        
        return fig
    
    def create_payment_method_chart(self, df):
        """Create payment method distribution chart"""
        payment_counts = df['Payment Method'].dropna().value_counts()
        
        fig = px.pie(
            values=payment_counts.values,
            names=payment_counts.index,
            title="Payment Method Distribution",
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def create_monthly_trend_chart(self, df):
        """Create monthly booking trend chart"""
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy['YearMonth'] = df_copy['Date'].dt.to_period('M')
        
        monthly_counts = df_copy.groupby('YearMonth').size()
        monthly_completion = df_copy.groupby('YearMonth')['Booking Status'].apply(
            lambda x: (x == 'Completed').sum()
        )
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add booking counts
        fig.add_trace(
            go.Scatter(
                x=monthly_counts.index.astype(str),
                y=monthly_counts.values,
                name="Total Bookings",
                line=dict(color='blue')
            ),
            secondary_y=False,
        )
        
        # Add completion counts
        fig.add_trace(
            go.Scatter(
                x=monthly_completion.index.astype(str),
                y=monthly_completion.values,
                name="Completed Rides",
                line=dict(color='green')
            ),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Total Bookings", secondary_y=False)
        fig.update_yaxes(title_text="Completed Rides", secondary_y=True)
        
        fig.update_layout(title_text="Monthly Booking Trends")
        
        return fig
    
    def create_daily_pattern_chart(self, df):
        """Create daily pattern chart (day of week)"""
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy['DayOfWeek'] = df_copy['Date'].dt.day_name()
        
        # Define the order of days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = df_copy['DayOfWeek'].value_counts().reindex(day_order)
        
        fig = px.bar(
            x=daily_counts.index,
            y=daily_counts.values,
            title="Daily Booking Pattern (Day of Week)",
            color=daily_counts.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Number of Bookings",
            showlegend=False
        )
        
        return fig
    
    def create_booking_heatmap(self, df):
        """Create booking heatmap for hour vs day of week"""
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy['Time'] = pd.to_datetime(df_copy['Time'], format='%H:%M:%S')
        df_copy['Hour'] = df_copy['Time'].dt.hour
        df_copy['DayOfWeek'] = df_copy['Date'].dt.day_name()
        
        # Create heatmap data
        heatmap_data = df_copy.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            title="Booking Heatmap: Hour vs Day of Week",
            color_continuous_scale='Blues',
            aspect='auto'
        )
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week"
        )
        
        return fig
    
    def create_price_distribution_chart(self, df):
        """Create price distribution chart"""
        completed_rides = df[(df['Booking Status'] == 'Completed') & (df['Booking Value'].notna())]
        
        if completed_rides.empty:
            return go.Figure().add_annotation(text="No completed rides data available")
        
        fig = px.histogram(
            completed_rides,
            x='Booking Value',
            nbins=50,
            title="Booking Value Distribution (Completed Rides)",
            color_discrete_sequence=['skyblue']
        )
        
        fig.update_layout(
            xaxis_title="Booking Value (₹)",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
    
    def create_distance_vs_price_scatter(self, df):
        """Create scatter plot for distance vs price"""
        completed_rides = df[
            (df['Booking Status'] == 'Completed') & 
            (df['Booking Value'].notna()) & 
            (df['Ride Distance'].notna())
        ]
        
        if completed_rides.empty:
            return go.Figure().add_annotation(text="No data available for distance vs price analysis")
        
        fig = px.scatter(
            completed_rides,
            x='Ride Distance',
            y='Booking Value',
            color='Vehicle Type',
            title="Ride Distance vs Booking Value",
            trendline="ols",
            opacity=0.6
        )
        
        fig.update_layout(
            xaxis_title="Ride Distance (km)",
            yaxis_title="Booking Value (₹)"
        )
        
        return fig
    
    def create_cancellation_analysis_chart(self, df):
        """Create cancellation analysis chart"""
        # Calculate cancellation rates by different factors
        cancellation_data = []
        
        # By vehicle type
        for vehicle_type in df['Vehicle Type'].unique():
            vehicle_data = df[df['Vehicle Type'] == vehicle_type]
            total = len(vehicle_data)
            cancelled = len(vehicle_data[vehicle_data['Booking Status'].str.contains('Cancel', na=False)])
            rate = (cancelled / total * 100) if total > 0 else 0
            cancellation_data.append({
                'Category': 'Vehicle Type',
                'Subcategory': vehicle_type,
                'Cancellation_Rate': rate,
                'Total_Bookings': total
            })
        
        cancellation_df = pd.DataFrame(cancellation_data)
        
        fig = px.bar(
            cancellation_df,
            x='Subcategory',
            y='Cancellation_Rate',
            title="Cancellation Rate by Vehicle Type",
            color='Cancellation_Rate',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            xaxis_title="Vehicle Type",
            yaxis_title="Cancellation Rate (%)",
            showlegend=False
        )
        
        return fig
    
    def create_ratings_analysis_chart(self, df):
        """Create ratings analysis chart"""
        completed_rides = df[
            (df['Booking Status'] == 'Completed') & 
            (df['Driver Ratings'].notna()) & 
            (df['Customer Rating'].notna())
        ]
        
        if completed_rides.empty:
            return go.Figure().add_annotation(text="No ratings data available")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Driver Ratings Distribution", "Customer Ratings Distribution")
        )
        
        # Driver ratings
        fig.add_trace(
            go.Histogram(
                x=completed_rides['Driver Ratings'],
                name="Driver Ratings",
                nbinsx=20,
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Customer ratings
        fig.add_trace(
            go.Histogram(
                x=completed_rides['Customer Rating'],
                name="Customer Ratings",
                nbinsx=20,
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Ratings Distribution Analysis",
            showlegend=False
        )
        
        return fig
