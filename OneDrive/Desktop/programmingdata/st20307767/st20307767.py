import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import folium  
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Function to load and preprocess data
def data_handling(data_handling):
    data_handling_list = []
    for data_handling_file in data_handling:
        df = pd.read_csv(data_handling_file)
        data_handling_list.append(df)

    # Combine all datasets into one DataFrame
    data_handling_part_system = pd.concat(data_handling_list, ignore_index=True)

    # Handle missing values
    data_handling_part_system.ffill(inplace=True)  # Forward fill missing values

    # Remove duplicate entries
    data_handling_part_system.drop_duplicates(inplace=True)

    # Feature engineering (e.g., create 'Month' from 'year')
    if 'year' in data_handling_part_system.columns:
        data_handling_part_system['year'] = pd.to_datetime(data_handling_part_system['year'], errors='coerce')
        data_handling_part_system['Month'] = data_handling_part_system['year'].dt.month
        data_handling_part_system.dropna(subset=['year'], inplace=True)

    return data_handling_part_system

# Streamlit App
def main():
    st.title("Air Quality Data Analysis System Using Streamlit")
    st.sidebar.title("Options")

    # Upload CSV files
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        # Process uploaded files
        data = pd.concat([data_handling([uploaded_file]) for uploaded_file in uploaded_files], ignore_index=True)
        st.success("Datasets loaded and processed successfully!")
    else:
        st.warning("Please upload your CSV file.")

    # Sidebar options
    show_data = st.sidebar.checkbox("Show Data")
    summary_stats = st.sidebar.checkbox("Exploratory Data Analysis (EDA)")
    model_building = st.sidebar.checkbox("Machine Learning Model Building")
    model_evaluation = st.sidebar.checkbox("Model Evaluation")

    if 'data' in locals():
        st.subheader("Dataset Overview")
        rows, columns = data.shape
        st.write(f"The dataset contains *{rows} rows* and *{columns} columns*.")
        st.write("The columns in the dataset are:", data.columns.tolist())

        # Show data types and missing values
        st.write("\n*Data Types:*", data.dtypes)
        st.write("\n*Missing Values:*", data.isnull().sum())

        # Show raw data
        if show_data:
            st.subheader("Data Preview")
            st.dataframe(data.head())

        # Show summary statistics and visualizations
        if summary_stats:
            st.subheader("Statistical Summary")
            st.write(data.describe())

            st.subheader("Feature Visualizations")
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

            for column in numeric_columns:
                st.subheader(f"Visualization for {column}")

                # Histogram
                plt.figure(figsize=(10, 6))
                sns.histplot(data[column], kde=True, color='skyblue', bins=30)
                st.pyplot(plt)
                plt.close()

                # Box Plot
                plt.figure(figsize=(10, 6))
                sns.boxplot(data[column], color='lightgreen')
                st.pyplot(plt)
                plt.close()

        # Map for highest pollution levels
        st.subheader("Map: Highest Pollution Locations")
        if 'latitude' in data.columns and 'longitude' in data.columns:
            if 'PM2.5' in data.columns:
                highest_pollution_station = data.loc[data['PM2.5'].idxmax()]
                m = folium.Map(location=[highest_pollution_station['latitude'], highest_pollution_station['longitude']], zoom_start=10)
                marker_cluster = MarkerCluster().add_to(m)
                folium.Marker(
                    location=[highest_pollution_station['latitude'], highest_pollution_station['longitude']],
                    popup=f"Station: {highest_pollution_station.get('Station', 'Unknown')}<br>PM2.5: {highest_pollution_station['PM2.5']}",
                    icon=folium.Icon(color='red')
                ).add_to(marker_cluster)
                st.components.v1.html(m._repr_html_(), height=500)
            else:
                st.write("PM2.5 data is missing. Please check the dataset.")
        else:
            st.write("Latitude and Longitude data are missing. Please check the dataset.")

        # Machine Learning Model Building
        if model_building:
            st.subheader("Machine Learning Model")
            target_column = st.selectbox("Select Target Variable:", numeric_columns)

            # Preprocess data
            features = data.drop(columns=['year', 'Station', target_column], errors='ignore')
            target = data[target_column]

            data_clean = data.dropna(subset=[target_column])
            features_clean = data_clean.drop(columns=['year', 'Station', target_column], errors='ignore')
            target_clean = data_clean[target_column]

            # Encoding categorical features
            features_clean = pd.get_dummies(features_clean, drop_first=True)

            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            features_imputed = imputer.fit_transform(features_clean)

            # Feature scaling
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_imputed)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_clean, test_size=0.2, random_state=42)

            # Choose model
            model_type = st.selectbox("Select Model:", ["Linear Regression", "Random Forest"])
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(random_state=42)

            # Fit the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Model Evaluation
            if model_evaluation:
                st.subheader("Model Evaluation")
                st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
                st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
                st.write("R-Squared:", r2_score(y_test, y_pred))

if __name__ == '__main__':
    main()