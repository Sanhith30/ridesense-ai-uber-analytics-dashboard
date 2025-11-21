
# 🚕 **RideSense AI — Intelligent Uber Ride Analytics & Prediction Dashboard**

### *An end-to-end ride analytics, forecasting, and visualization system powered by Streamlit, Plotly, and Machine Learning.*

---

## 📊 **Overview**

**RidePredictor** is a complete data analytics and machine learning dashboard built to analyze and predict ride behavior using NCR ride data.  
It combines:

- 📈 Interactive analytics  
- 🤖 Multiple ML models  
- 🧹 Automated data processing  
- 🎨 Advanced visualizations  
- 🧠 Smart predictions (fare, duration, completion)

This project demonstrates:  
**Data → Feature Engineering → ML Modeling → Interactive Dashboard → Insights**

---

## 🚀 **Key Features**

### 🧠 Machine Learning Predictions  
- Ride completion probability  
- Estimated fare (₹)  
- Ride duration prediction  
- Cancellation risk insights  

### 📈 Interactive Dashboard Pages  
- Dashboard Overview  
- Predict Ride (ML)  
- Data Analytics  
- Model Performance  

### 🔧 Data Processing  
- Datetime parsing  
- Derived feature creation  
- Handling missing values  
- Location grouping  
- Encoding categorical variables  

### 🎨 Visual Analytics  
- Booking status distribution  
- Hourly/weekly/monthly trends  
- Vehicle performance  
- Price vs distance  
- Cancellation patterns  
- Heatmaps  

---

## 🗂️ **Project Structure**

```
RidePredictor/
│── __pycache__/
│
│── attached_assets/
│     ├── Dasboard_1755850616783.gif
│     ├── Uber_1755850616782.pbix
│     └── ncr_ride_bookings_1755850616781.csv
│
│── app.py
│── data_processor.py
│── ml_models.py
│── visualizations.py
│── utils.py
│── requirements.txt
│── pyproject.toml
│── replit.md
│── uv.lock
│── README.md
```

---

## 📂 **Files Overview**

### **`app.py`**  
Main Streamlit dashboard with navigation, predictions, charts, and analytics.

### **`data_processor.py`**  
- Data cleaning  
- Feature engineering  
- Date, time, ratings, distance processing  
- ML feature preparation  

### **`ml_models.py`**  
Trains ML models:  
- RandomForestClassifier (completion)  
- GradientBoostingRegressor (fare)  
- RandomForestRegressor (duration)  
Includes performance evaluation.

### **`visualizations.py`**  
Creates interactive visualizations using Plotly.

### **`utils.py`**  
Formatting utilities + summary metric computations.

### **`attached_assets/`**  
Contains dataset & GIF preview.

---

## 📸 **Dashboard Preview**

![Dashboard Preview](RidePredictor/attached_assets/Dasboard_1755850616783.gif)

---

## 🧬 **Tech Stack**

### Dashboard  
- Streamlit  
- Plotly  

### Machine Learning  
- Scikit-Learn  
- RandomForest  
- GradientBoosting  

### Data Handling  
- Pandas  
- NumPy  

---

## 📑 **Dataset**

Stored at:

```
attached_assets/ncr_ride_bookings_1755850616781.csv
```

Contains fields like:

- Date, Time  
- Booking Status  
- Distance  
- Fare  
- Ratings  
- Vehicle Type  
- Cancellation reasons  

---

## 📊 **Analytics Provided**

- Temporal ride trends  
- Route analysis  
- Pickup/drop hotspots  
- Cancellation breakdown  
- Distance vs fare regression  
- Vehicle type comparisons  

---

## 🧱 **Why This Project Stands Out**

✔ Clean modular code  
✔ Real-world ML pipeline  
✔ Full interactive visualization  
✔ Recruiter-ready project  
✔ Production-style structure  

---

## 🙌 **Author**

**🧑‍💻 Thikkavarapu Sanhith**  
Data Analyst | ML Engineer | AI Builder

---

## ⭐ **Please Star the Repository**

Your support motivates more open-source work!

