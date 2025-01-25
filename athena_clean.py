import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import scipy.stats as stats


st.image("rtds.jpg", width =700)

st.title("Athena Algorithm", help = "What is Athena? Athena is a set of data processing rules. Imagine it like a large flow chart full of math. If you have time-series data, like a data log, Athena helps you process it into a simple OKAY or NOT OKAY output. You only have to supply the exposure limit(s) and the units of measure.")
st.markdown((""" Created by [Caleb Ginorio](https://www.linkedin.com/in/caleb-ginorio-58b243b4/), [Steve Jahn](https://www.linkedin.com/in/steven-jahn-mba-cih-faiha-85a04871/) & [Spencer Pizzani](https://www.linkedin.com/in/spencer-pizzani-cih-232a7518/)"""))

st.subheader("Why should I use Athena?")

with st.expander("Click to Expand"):
    tab1, tab2, tab3 = st.tabs(["Long Answer", "Short Answer", "Really Short Answer"])

    with tab1:
        st.markdown(("[The Athena heuristic: The need for a system of algorithms for standardized evaluation of big exposure data](https://www.tandfonline.com/doi/abs/10.1080/15459624.2022.2132259?journalCode=uoeh20&)"))

    with tab2:
        st.markdown("""
                - As the amount of data collected by direct-reading instruments grows, it becomes difficult to manage. Many OEHS professionals are still comparing averages in spreadsheets directly to exposure limits. We know we need to do better, but doing better is difficult. It’s especially challenging in situations where resources are scarce.
                - It also takes years to build enough experience to be able to analyze data at a level needed to use all of our best practices. Instead of training everyone to get really good at data analysis, we’re making it easier to do by building consensus on how it should be done. Athena is composed of those rules.
                - We’ve done this before. The Brief and Scala (1986) model used in Athena is more accessible than more complex processes to adjust exposure limits for extended shifts. Sometimes a simple and easy to get answer is better than a precise answer that is beyond reach, or there’s just too much data to analyze.""")
        
    with tab3:
        st.image("athenameme.jpg")


# molecular weights file
mw_df = pd.read_excel('ChemicalMolecularWeight.xlsx')
mw_df.set_index('ChemicalName', inplace=True)


st.divider()

with st.sidebar:
    st.header("Limit Values")

    st.subheader("Time-Weighted Average (TWA)",
                  help= "The TWA concentration for a conventional 8-hour workday and a 40-hour workweek,to which it is believed that nearly all workers may be repeatedly exposed, day after day, for a working lifetime without adverse effect.")
    # input for the 8-hour TWA
    twa_8hr_limit = st.number_input('Enter the 8-hour TWA limit:', min_value=0.0, format="%.2f")
    twa_10hr_limit = twa_8hr_limit * 0.7
    twa_12hr_limit = twa_8hr_limit * 0.5

    st.subheader("STEL", help ="A 15-minute TWA exposure that should not be exceeded at any time during a workday, even if the 8-hour TWA is within the TLV–TWA. The TLV–STEL is the concentration to which it is believed that nearly all workers can be exposed continuously for a short period of time without suffering from 1) irritation, 2) chronic or irreversible tissue damage, 3) dose-rate-dependent toxic effects, or 4) narcosis of sufficient degree to increase the likelihood of accidental injury, impaired self-rescue, or materially reduced work efficiency.")
    use_stel = st.checkbox('Specify STEL')
    stel_limit = None
    if use_stel:
        stel_limit = st.number_input('Enter the STEL:', min_value=0.01, format="%.2f", help="Specify the Short-Term Exposure Limit.")
        time_interval = st.selectbox(
            'Select the time interval:',
            options=['15-Minute', '30-Minute', '2-Hour'],
            index=0  # Default selection is '15-Minute'
        )

    st.subheader("Ceiling", help="The concentration that should not be exceeded during any part of the working exposure. If instantaneous measurements are not available, sampling should be conducted for the minimum period of time sufficient to detect exposures at or above the ceiling value.")
    use_ceiling = st.checkbox('Specify Ceiling Limit')
    ceiling_limit = None
    if use_ceiling:
        ceiling_limit = st.number_input('Enter the ceiling limit:', min_value=0.01, format="%.2f")


    st.header("Unit of Measure")
    current_unit = st.selectbox("Current unit of the concentration column", ['PPM', 'Percent Volume (%)', 'mg/m^3', 'micrograms/m^3'])

    convert_units = st.checkbox("Convert units")

    desired_unit = None
    if convert_units:
        desired_unit = st.selectbox("Convert to", ['PPM', 'Percent Volume (%)', 'mg/m^3', 'micrograms/m^3'])


st.header('File Upload', help = "Make sure your file starts with data headers (column titles) in the first row, and that the time-series data is in columns beneath those headers.")

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['xlsx', 'csv'])

def load_data(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type == 'xlsx':
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            
            if len(sheet_names) > 1:
                sheet_selection_expander = st.expander("Select Excel Sheet")
                with sheet_selection_expander:
                    selected_sheet = st.selectbox("Select the sheet", sheet_names)
                    return pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            else:
                return pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
        elif file_type == 'csv':
            return pd.read_csv(uploaded_file)

df = load_data(uploaded_file)

st.divider()

if df is not None and not df.empty:
    st.header("Data Selection")
    with st.expander("Specify Data Columns"):
        date_time_format = st.radio(
            "Is the date and time information in the same column or separate columns?",
            ('Single Column', 'Separate Columns'))

        if date_time_format == 'Single Column':
            datetime_col = st.selectbox("Select the Date/Time Column", options=df.columns)
        elif date_time_format == 'Separate Columns':
            date_col = st.selectbox("Select the Date Column", options=df.columns, index=0)
            time_col = st.selectbox("Select the Time Column", options=df.columns, index=1)
        
        concentration_col = st.selectbox("Select the Concentration Column", options=df.columns)

        if st.button("Show Selected Columns Data"):
            if date_time_format == 'Single Column':
                selected_data = df[[datetime_col, concentration_col]]
            else:
                selected_data = df[[date_col, time_col, concentration_col]]
            st.write(selected_data.head())

    st.divider()
    st.subheader("Quality Control Check")
    perform_qc = st.checkbox("Perform QC Check")
    if perform_qc:
        missing = df[concentration_col].isna()
        negatives = df[concentration_col] < 0

        if missing.any() or negatives.any():
            st.error("QC Alert: There are missing or negative values.")
            if missing.any():
                st.write(f"Missing values count: {missing.sum()}")
            if negatives.any():
                st.write(f"Negative values count: {(negatives).sum()}")

            remove_issues = st.checkbox("Remove Missing and Negative Values")
            if remove_issues:
                df = df[~missing & ~negatives].copy()
                st.session_state.df = df
                st.success("Problematic rows removed.")
        else:
            st.success("No issues detected in the selected data column.")
    
    st.divider()

    if date_time_format == 'Single Column':
        df['DateTime'] = pd.to_datetime(df[datetime_col])
    else:
        df['DateTime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col])
    df.set_index('DateTime', inplace=True)

    supported_conversions = {
        ('Percent Volume (%)', 'PPM'): lambda x: x * 10000,
        ('PPM', 'Percent Volume (%)'): lambda x: x / 10000,
        ('PPM', 'mg/m^3'): lambda x, mw: x * (mw * 0.0409), 
        ('mg/m^3', 'PPM'): lambda x, mw: x / (mw * 0.0409),
        ('PPM', 'micrograms/m^3'): lambda x, mw: x * (mw * 40.9),
        ('micrograms/m^3', 'PPM'): lambda x, mw: x / (mw * 40.9),
        ('mg/m^3', 'micrograms/m^3'): lambda x: x * 1000,
        ('micrograms/m^3', 'mg/m^3'): lambda x: x / 1000,
        ('Percent Volume (%)', 'mg/m^3'): lambda x, mw: x * 10000 * (mw * 0.0409),
        ('mg/m^3', 'Percent Volume (%)'): lambda x, mw: x / (10000 * (mw * 0.0409)),
        ('Percent Volume (%)', 'micrograms/m^3'): lambda x, mw: x * 10000 * (mw * 40.9),
        ('micrograms/m^3', 'Percent Volume (%)'): lambda x, mw: x / (10000 * (mw * 40.9))
    }

    if convert_units and desired_unit and current_unit != desired_unit:
        if (current_unit, desired_unit) in supported_conversions:
            if 'mw' in supported_conversions[(current_unit, desired_unit)].__code__.co_varnames:
                chemical = st.sidebar.selectbox("Select The Chemical Agent", mw_df.index, help ="Athena knows the molecular weight of these chemicals, so we can convert from parts per million (ppm) or milligrams per cubic meter (mg/m3) for you. The unit of measure of your data, or the unit of measure in the “Convert to” field, must be the same as the unit of measure used for your exposure limits.")
                molecular_weight = mw_df.loc[chemical, 'MolecularWeight']
                df['converted'] = supported_conversions[(current_unit, desired_unit)](df[concentration_col], molecular_weight)
            else:
                df['converted'] = supported_conversions[(current_unit, desired_unit)](df[concentration_col])
        else:
            st.error("Conversion from {} to {} is not supported.".format(current_unit, desired_unit))
            st.stop() 
    else:
        df['converted'] = df[concentration_col]
        desired_unit = current_unit

    unit_display = desired_unit if convert_units else current_unit

    st.subheader("Data for Analysis")
    with st.expander(f"View Data in {unit_display}"):
        st.write(df[['converted']])

    st.divider()

    # rolling TWA calculation
 
    df['RollingTWA_8hr'] = df['converted'].rolling(window=8*60, min_periods=1).mean()
    df['RollingTWA_10hr'] = df['converted'].rolling(window=10*60, min_periods=1).mean()
    df['RollingTWA_12hr'] = df['converted'].rolling(window=12*60, min_periods=1).mean()


    if use_stel:
        stel_minutes = int(time_interval.split('-')[0]) * 60
        df['RollingSTEL'] = df['converted'].rolling(window=stel_minutes, min_periods=1).mean()

    st.header("Time Series", help ="This graph compares your concentrations to the exposure limits you entered, so you can visualize how much of the day you are over (or under) that limit. ")
    
    # plotting

    fig = px.line(df, x=df.index, y='converted', labels={'converted': f'Concentration ({unit_display})'}, title=f'Time Series of Concentration ({unit_display})')
    fig.add_scatter(x=df.index, y=df['RollingTWA_8hr'], mode='lines', name='8-Hr Rolling TWA')
    fig.add_scatter(x=df.index, y=df['RollingTWA_10hr'], mode='lines', name='10-Hr Rolling TWA')
    fig.add_scatter(x=df.index, y=df['RollingTWA_12hr'], mode='lines', name='12-Hr Rolling TWA')
    fig.add_hline(y=twa_8hr_limit, line_dash="dash", line_color="blue", annotation_text="8hr TWA Limit")
    fig.add_hline(y=twa_10hr_limit, line_dash="dash", line_color="green", annotation_text="10hr TWA Limit")
    fig.add_hline(y=twa_12hr_limit, line_dash="dash", line_color="red", annotation_text="12hr TWA Limit")
   

    if use_stel and stel_limit:
        fig.add_scatter(x=df.index, y=df['RollingSTEL'], mode='lines', name='STEL')
        fig.add_hline(y=stel_limit, line_dash="dash", line_color='orange', annotation_text="STEL Limit")
    if use_ceiling and ceiling_limit:
        fig.add_hline(y=ceiling_limit, line_dash="dash", line_color='red', annotation_text="Ceiling Limit")
    
    st.plotly_chart(fig)
   

    st.divider()

    st.subheader("Analysis Outcome", help ="This is the result of the Athena analysis. It will never change for a given set of data. Not exceeding the exposure limit is not the same as being safe. Please manage occupational exposures carefully. This process is not a substitute for professional practice.")
    # check for exceedances
    df['TWA_8hr_Exceeded'] = df['RollingTWA_8hr'] > twa_8hr_limit
    df['TWA_10hr_Exceeded'] = df['RollingTWA_10hr'] > twa_10hr_limit
    df['TWA_12hr_Exceeded'] = df['RollingTWA_12hr'] > twa_12hr_limit

    df['STEL_Exceeded'] = df['RollingSTEL'] > stel_limit if use_stel and stel_limit else False
    df['Ceiling_Exceeded'] = df['converted'] > ceiling_limit if use_ceiling and ceiling_limit else False

    col1, col2, col3, col4, col5= st.columns(5)
    with col1:
        st.write("8-hour TWA:")
        if df['TWA_8hr_Exceeded'].any():
            st.markdown(":thumbsdown:", unsafe_allow_html=True)
            st.markdown("<span style='color: red;'>Exceeded</span>", unsafe_allow_html=True)
        else:
            st.markdown(":thumbsup:", unsafe_allow_html=True)
            st.markdown("<span style='color: green;'>Not Exceeded</span>", unsafe_allow_html=True)


    with col3:
        st.write("10-hour TWA:")
        if df['TWA_10hr_Exceeded'].any():
            st.markdown(":thumbsdown:", unsafe_allow_html=True)
            st.markdown("<span style='color: red;'>Exceeded</span>", unsafe_allow_html=True)
        else:
            st.markdown(":thumbsup:", unsafe_allow_html=True)
            st.markdown("<span style='color: green;'>Not Exceeded</span>", unsafe_allow_html=True)


    with col5:
        st.write("12-hour TWA:")
        if df['TWA_12hr_Exceeded'].any():
            st.markdown(":thumbsdown:", unsafe_allow_html=True)
            st.markdown("<span style='color: red;'>Exceeded</span>", unsafe_allow_html=True)
        else:
            st.markdown(":thumbsup:", unsafe_allow_html=True)
            st.markdown("<span style='color: green;'>Not Exceeded</span>", unsafe_allow_html=True)

    
    col6, col7, col8, col9, col10,= st.columns(5)
    if use_stel:
        with col7:
            st.write("STEL:")
            if df['STEL_Exceeded'].any():
                st.markdown(":thumbsdown:", unsafe_allow_html=True)
                st.markdown("<span style='color: red;'>Exceeded</span>", unsafe_allow_html=True)
            else:
                st.markdown(":thumbsup:", unsafe_allow_html=True)
                st.markdown("<span style='color: green;'>Not Exceeded</span>", unsafe_allow_html=True)


    if use_ceiling:
        with col9:
            st.write("Ceiling:")
            if df['Ceiling_Exceeded'].any():
                st.markdown(":thumbsdown:", unsafe_allow_html=True)
                st.markdown("<span style='color: red;'>Exceeded</span>", unsafe_allow_html=True)
            else:
                st.markdown(":thumbsup:", unsafe_allow_html=True)
                st.markdown("<span style='color: green;'>Not Exceeded</span>", unsafe_allow_html=True)


    st.divider()

#    st.subheader("Log-Normality Test", help = "Some analysis assumes that occupational exposures are lognormal. This process tests to see if the data you uploaded looks lognormal, as far as we can tell. If your data isn’t lognormal, the Bayesian analysis may not work as intended. If your data contains non-positive values, this part won’t work (for math reasons, not programming problems).")

#    if st.checkbox('Check if data is log-normally distributed'):
        # ensure data is positive since log transformation requires positive values
#        if (df['converted'] <= 0).any():
#            st.error("Data contains non-positive values which cannot be log-transformed.")
#        else:
#            log_data = np.log(df['converted'])

#            # histogram of the log-transformed data
#            fig_hist = px.histogram(log_data, nbins=30, title="Log-Transformed Data Histogram")
#            fig_hist.update_layout(bargap=0.1)
#            st.plotly_chart(fig_hist)

#            # Q-Q plots
#            norm_quantiles = np.linspace(0, 1, num=len(log_data))
#            data_quantiles = np.quantile(log_data, norm_quantiles)
#            theoretical_quantiles = stats.norm.ppf(norm_quantiles, np.mean(log_data), np.std(log_data))

#            qq_fig = go.Figure()
#            qq_fig.add_trace(go.Scatter(x=theoretical_quantiles, y=data_quantiles, mode='markers',
#                                        name='Data Quantiles'))
#            qq_fig.add_trace(go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles, mode='lines',
#                                        name='Theoretical Quantiles'))
#            qq_fig.update_layout(title="Q-Q Plot of Log-Transformed Data",
#                                xaxis_title='Theoretical Quantiles',
#                                yaxis_title='Sample Quantiles')
#            st.plotly_chart(qq_fig)

#            # normality test (Shapiro-Wilk test)
#            stat, p_value = stats.shapiro(log_data)
#            st.write("Shapiro-Wilk Test Results:")
#            st.write("Statistic:", stat, "P-value:", p_value)

#            # interpretation
#            alpha = 0.05  # significance level
#            if p_value > alpha:
#                st.success('Data looks log-normal (fail to reject H0 at alpha = 0.05)')
#            else:
#                st.error('Data does not look log-normal (reject H0 at alpha = 0.05)')


st.sidebar.divider()

st.sidebar.header("Real-Time Detection Systems (RTDS) Volunteer Committee")
st.sidebar.markdown(( """[Joins Us!](https://www.aiha.org/get-involved/volunteer-groups/real-time-detection-systems-committee)"""))


st.sidebar.markdown(("""[Donate](https://www.aiha.org/get-involved/aih-foundation/aih-foundation-scholarships/real-time-detection-systems-scholarship) today to the RTDS Scholarship!"""))



