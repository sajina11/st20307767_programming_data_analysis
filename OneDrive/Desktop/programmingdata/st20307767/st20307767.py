
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
# Streamlit App
def main():
    st.title(" Air Quality Data Analysis system Using streamlit")
    st.sidebar.title("Clicking the option value ")

    # Upload CSV files
    uploaded_files = st.sidebar.file_uploader("Upload CSV files according to user like ", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        # Process uploaded files
        data = pd.concat([data_handling([uploaded_file]) for uploaded_file in uploaded_files], ignore_index=True)
        st.success("Datasets loaded and processed successfully!")
    else:
        st.warning("Upload your CSV file")

    # Sidebar options
    show_data = st.sidebar.checkbox("Data Handling")
    summary_stats = st.sidebar.checkbox("Exploratory Data Analysis (EDA)")
    model_building = st.sidebar.checkbox("Machine Learning Model Building")
    model_evaluation = st.sidebar.checkbox("Model Evaluation")

    # Show dataset insights
    if 'data' in locals():
        st.subheader("Dataset ")
        rows, columns = data.shape
        st.write(f"The dataset contains *{rows} rows* and *{columns} columns*.")
        st.write("The columns in the dataset are:")
        st.write(data.columns.tolist())

        # Show data types and check for missing values
        st.write("\n*Data Types of Each Column:*")
        st.write(data.dtypes)

        missing_values = data.isnull().sum()
        st.write("\n*Missing Values:*")
        st.write(missing_values)

        # Show raw data
        if show_data:
            st.subheader("Initial Data set")
            st.dataframe(data.head())

        # Show summary statistics
        if summary_stats:
            st.subheader(" Show all the value of Statistics For dataset")
            st.write(data.describe())

            # Display graphs for each numeric column
            st.subheader("Visualizations Features Of air quality")
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

            for column in numeric_columns:
                st.subheader(f"Visualization for {column}")
                plt.figure(figsize=(10, 6))
                sns.histplot(data[column], kde=True, color='skyblue', bins=30)
                st.pyplot(plt)

                # Box Plot
                plt.figure(figsize=(10, 6))
                sns.boxplot(data[column], color='lightgreen')
                st.pyplot(plt)

        # Map for highest pollution levels
        st.subheader("Map: Highest Pollution Locations")
        if 'latitude' in data.columns and 'longitude' in data.columns:
            if 'PM2.5' in data.columns:
                highest_pollution_station = data.loc[data['PM2.5'].idxmax()]
                m = folium.Map(location=[39.9042, 116.4074], zoom_start=10)
                marker_cluster = MarkerCluster().add_to(m)
                folium.Marker(
                    location=[highest_pollution_station['latitude'], highest_pollution_station['longitude']],
                    popup=f"Station: {highest_pollution_station.get('Station', 'Unknown')}<br>PM2.5: {highest_pollution_station['PM2.5']}",
                    icon=folium.Icon(color='red')
                ).add_to(marker_cluster)
                st.write(" Highest pollution is displayed")
                st.components.v1.html(m.repr_html(), height=500)
            else:
                st.write("PM2.5 data is missing in the dataset. Please check in the CSV file.")
        else:
            st.write("Latitude and Longitude data are missing. Please check in the CSV dataset.")

        # Machine Learning Model Building
        if model_building:
            st.subheader("Implementing the Machine learning method")
            target_column = st.selectbox("Select  Variable:", numeric_columns)

            features = data.drop(columns=['year', 'Station', target_column], errors='ignore')
            target = data[target_column]

            # Remove rows where the target variable contains NaN
            data_clean = data.dropna(subset=[target_column])
            features_clean = data_clean.drop(columns=['year', 'Station', target_column], errors='ignore')
            target_clean = data_clean[target_column]

            # Encoding categorical features (if any)
            features_clean = pd.get_dummies(features_clean, drop_first=True)

            # Handle missing values using SimpleImputer
            imputer = SimpleImputer(strategy='mean')  # Use 'mean' for imputation
            features_imputed = imputer.fit_transform(features_clean)

            # Feature scaling
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_imputed)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_clean, test_size=0.2, random_state=42)

            # Choose model
            model_type = st.selectbox("Select Method for machine learning Model:", ["Linear Regression", "Random Forest"])
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(random_state=42)

            # Fit the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluation
            if model_evaluation:
                st.subheader("Model Evaluation")
                st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
                st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
                st.write("R-Squared:", r2_score(y_test, y_pred))

if __name__ == '__main__':
    main()
