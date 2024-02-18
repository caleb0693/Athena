import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

img = "rtds.jpg"
st.image(img, width =600)

# Streamlit app heading
st.title("Athena File Processor")
st.write("Created by Caleb Ginorio,  Steve Jahn, &  Spencer Pizzani")

st.markdown("""
 * Use the menu at left to select occupational exposure limit (OEL)
 * Your plots and analysis will appear below
""")

st.divider()

st.sidebar.header("TWA Input")

# User input for the 8-hour TWA limit
twa_8hr_limit = st.sidebar.number_input('Enter the 8-hour TWA limit (PPM):', min_value=0.0, format="%.2f")
twa_10hr_limit = twa_8hr_limit * 0.7
twa_12hr_limit = twa_8hr_limit * 0.5

st.sidebar.write(f"10-hour TWA limit: {twa_10hr_limit:.2f} PPM")
st.sidebar.write(f"12-hour TWA limit: {twa_12hr_limit:.2f} PPM")


st.sidebar.header("Real-Time Detection Systems (RTDS) Volunteer Committee")
st.sidebar.markdown(( """[Joins Us!](https://www.aiha.org/get-involved/volunteer-groups/real-time-detection-systems-committee)"""))


st.sidebar.markdown(("""[Donate](https://www.aiha.org/get-involved/aih-foundation/aih-foundation-scholarships/real-time-detection-systems-scholarship) today to the RTDS Scholarship!"""))

# rolling TWA
def calculate_rolling_twa(df, column, window_hours):
    window_minutes = window_hours * 60  # Convert hours to minutes
    df[f'RollingTWA_{window_hours}hr'] = df[column].rolling(window=window_minutes, min_periods=1).mean()
    return df

# upload a file
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # to read the file
    df = pd.read_excel(uploaded_file)

    # to convert 'Value' column to PPM
    df["PPM"] = df["Value"] * 10000

    # histogram of PPM values
    fig_histogram = px.histogram(df, x="PPM", nbins=50, title= "Distribution of CO<sub>2</sub> Concentration")

    fig_histogram.update_xaxes(title_text=' CO<sub>2</sub> Concentration (PPM)')
    fig_histogram.update_yaxes(title_text='Frequency')
    st.plotly_chart(fig_histogram)

    #missing values
    missing_values_count = df["PPM"].isna().sum()
    total_values_count = df.shape[0]
    missing_percentage = (missing_values_count / total_values_count) * 100

     # to check for values under 400 PPM
    count_under_400 = df[df["PPM"] < 400].shape[0]
    if count_under_400 > 0:
        st.warning(f"Warning: {count_under_400} readings are under 400 PPM.")
    else:
        st.success("No readings are under 400 PPM.")

    # to check for negative values
    negative_count = df[df["PPM"] < 0].shape[0]
    if negative_count > 0:
        st.warning(f"Warning: {negative_count} negative readings detected.")
    else:
        st.success("No negative readings detected.")

    # to check for percentage of values <= 100 PPM
    low_value_percentage = (df[df["PPM"] <= 100].shape[0] / df.shape[0]) * 100
    if low_value_percentage > 0:
        st.warning(f"Warning: {low_value_percentage:.2f}% of readings are below 100 PPM.")
    else:
        st.success("No readings are below 100 PPM.")

    # Percentage of missing values if any
    if missing_values_count > 0:
        st.warning(f"Warning: There are missing values in the concentration column ({missing_percentage:.2f}% of total readings).")

        # Provide options for handling missing values
        missing_value_option = st.selectbox(
            "Choose how to handle missing values:",
            ("Remove rows with missing values", "Replace with mean", "Replace with median")
        )

        # Handling if missing values based on user selection
        if missing_value_option == "Remove rows with missing values":
            df = df.dropna(subset=["PPM"])
            st.success("Rows with missing values have been removed.")
        elif missing_value_option == "Replace with mean":
            mean_value = df["PPM"].mean()
            df["PPM"].fillna(mean_value, inplace=True)
            st.success(f"Missing values have been replaced with the mean: {mean_value:.2f}")
        elif missing_value_option == "Replace with median":
            median_value = df["PPM"].median()
            df["PPM"].fillna(median_value, inplace=True)
            st.success(f"Missing values have been replaced with the median: {median_value:.2f}")
    else:
        st.success("There are no missing values in the dataset")


    st.divider()

    # to convert ReadTime to datetime
    df["ReadTime"] = pd.to_datetime(df["ReadTime"])
    df.sort_values("ReadTime", inplace=True)

    # Calculate rolling TWAs
    df['RollingTWA_8hr'] = df['PPM'].rolling(window=480, min_periods=1).mean()
    df['RollingTWA_10hr'] = df['PPM'].rolling(window=600, min_periods=1).mean()
    df['RollingTWA_12hr'] = df['PPM'].rolling(window=720, min_periods=1).mean()

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ReadTime"], y=df["PPM"], name="CO<sub>2</sub> Concentration", mode='lines', line=dict(color='black', width = 1.3)))
    fig.add_trace(go.Scatter(x=df["ReadTime"], y=df["RollingTWA_8hr"], name="8-Hr Rolling TWA", mode='lines', line=dict(color='red', width = 3)))
    fig.add_trace(go.Scatter(x=df["ReadTime"], y=df["RollingTWA_10hr"], name="10-Hr Rolling TWA", mode='lines', line=dict(color='blue', width = 2)))
    fig.add_trace(go.Scatter(x=df["ReadTime"], y=df["RollingTWA_12hr"], name="12-Hr Rolling TWA", mode='lines', line=dict(color='purple', width =2)))

    fig.add_hline(y=twa_8hr_limit, line_dash="dash", line_color="red", annotation_text='8hr TWA Limit')
    fig.add_hline(y=twa_10hr_limit, line_dash="dash", line_color="blue", annotation_text='10hr TWA Limit')
    fig.add_hline(y=twa_12hr_limit, line_dash="dash", line_color="purple", annotation_text='12hr TWA Limit')
    fig.update_yaxes(title_text= "CO<sub>2</sub> Concentration (PPM)")
  

    fig.update_layout(title="CO<sub>2</sub>  Concentration (PPM) and Rolling TWAs Over Time", xaxis_title="Time", yaxis_title="PPM", plot_bgcolor='white')
    fig.update_yaxes(title_text= "CO<sub>2</sub> Concentration (PPM)")
    st.plotly_chart(fig)

    st.divider()

    # Metrics for TWA exceedances
    exceedances = {
        "8hr TWA": np.any(df["RollingTWA_8hr"] > twa_8hr_limit),
        "10hr TWA": np.any(df["RollingTWA_10hr"] > twa_10hr_limit),
        "12hr TWA": np.any(df["RollingTWA_12hr"] > twa_12hr_limit),
    }

    col1, col2, col3 = st.columns(3)

    
    columns = {
    "8hr TWA": col1,
    "10hr TWA": col2,
    "12hr TWA": col3
    }
    
    for twa, exceeded in exceedances.items():
        with columns[twa]: 
            if exceeded:
                st.metric(label=f"{twa} Limit", value="Exceeded", delta="Consider actions", delta_color="inverse")
            else:
                st.metric(label=f"{twa} Limit", value="Not Exceeded", delta="Within limits", delta_color="normal")

