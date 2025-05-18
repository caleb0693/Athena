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
                - As the amount of data collected by direct-reading instruments grows, it becomes difficult to manage. Many OEHS professionals are still comparing averages in spreadsheets directly to exposure limits. We know we need to do better, but doing better is difficult. It’s especially in situations where resources are scarce.
                - It also takes years to build enough experience to be able to analyze data at a level needed to use all of our best practices. Instead of training everyone to get really good at data analysis, we’re making it easier to do by building consensus on how it should be done. Athena is composed of those rules.
                - We’ve done this before. The Brief and Scala (1986) model used in Athena is more accessible than more complex processes to adjust exposure limits for extended shifts. Sometimes a simple and easy to get answer is better than a precise answer that is beyond reach, or there’s just too much data to analyze.""")
        
    with tab3:
        st.image("athenameme.jpg")



mw_df = pd.read_excel('ChemicalMolecularWeight.xlsx')
mw_df.set_index('ChemicalName', inplace=True)


st.divider()

with st.sidebar:
    st.header("Limit Values")

    st.subheader("Time-Weighted Average (TWA)",
                  help= "The TWA concentration for a conventional 8-hour workday and a 40-hour workweek,to which it is believed that nearly all workers may be repeatedly exposed, day after day, for a working lifetime without adverse effect.")
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
            index=0 
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
        zeroes = df[concentration_col] == 0
        cleaning_method_used = None

        if missing.any() or negatives.any() or zeroes.any():
            st.error("QC Alert: There are missing, negative, or zero values.")
            st.write(f"Missing: {missing.sum()}, Negative: {negatives.sum()}, Zero: {zeroes.sum()}")

            st.subheader("Cleaning Options")
            with st.expander("What do these cleaning options do?"):
                st.markdown("""
                **Remove Negative and Missing Values** 
                - This function identifies negative concentrations or missing values (`NaN`) in the time-series data and replaces those values with
                                zeroes. This function assumes concentrations cannot be
                                negative and that missing values are a result of errors in the
                                data logging process. Best used when an instrument has no
                                filter band below the limit of quantification (LOQ).
                                Significant numbers of negative or missing values indicate a
                                calibration issue.
                            
                **Replace Negative and Missing Values with Average of Adjacent Nonzero Rows**
                - This function replaces removed negative and missing values (`NaN`) from the previous function with an average of the previous and next nonzero row. 
                            
                    For example:
                        
                        ```
                        10:00: 200
                        10:01: (Negative or Missing Value)
                        10:02: (Negative or Missing Value)
                        10:03: 300
                

                    This option will replace the zero in 10:01 and 10:02 with 250.

                    This function follows similar assumptions to flow rate calculations on an integrated (pump and tube) sample.

                            
                **Replace Zeroes and/or Missing Values with Average of Adjacent Nonzero Rows**
                - **Use with caution!** 

                    This function takes the previous and next nonzero measurement or a missing value and averages them, then replaces each zero row with that average. 
                    
                    For example:
                    ```
                    11:00: 100
                    11:01: 0
                    11:02: 0
                    11:03: 200
                    ```
                    This option will replace the zero in 11:01 and 11:02 with 150.

                    Depending on the context of your exposure, rows with zeroes may indicate an issue with a filter band. You may also want to adjust for the effect of workers leaving the area of exposure to make the TWA calculation more conservative. Best used for agents that are always present (CO2, O2).
                            When combined with Remove Negative and Missing Values, this function may be significantly more conservative than actual exposures.

                """)

            st.markdown("### Select Cleaning Method(s)")
            remove_neg_missing = st.checkbox("Remove Negative and Missing Values (Default On)", value=True)
            replace_neg_missing = st.checkbox("Replace Negative and Missing Values with Average of Adjacent Nonzero Rows")
            replace_zero_missing = st.checkbox("Replace Zeroes and/or Missing Values with Average of Adjacent Nonzero Rows")

            def replace_with_adjacent_avg(values, condition_fn):
                values = values.copy()
                for i in range(len(values)):
                    if condition_fn(values[i]):
                        prev_vals = values[:i][values[:i] != 0]
                        next_vals = values[i+1:][values[i+1:] != 0]
                        prev = prev_vals[-1] if len(prev_vals) > 0 else np.nan
                        next = next_vals[0] if len(next_vals) > 0 else np.nan
                        values[i] = np.nanmean([prev, next])
                return values

            if remove_neg_missing:
                df = df[~(missing | negatives)].copy()
                cleaning_method_used = "Removed negative and missing values"

            if replace_neg_missing:
                df[concentration_col] = replace_with_adjacent_avg(
                    df[concentration_col].values,
                    lambda x: pd.isna(x) or x < 0
                )
                cleaning_method_used = "Replaced negative and missing values with average of adjacent nonzero rows"

            if replace_zero_missing:
                df[concentration_col] = replace_with_adjacent_avg(
                    df[concentration_col].values,
                    lambda x: pd.isna(x) or x == 0
                )
                cleaning_method_used = "Replaced zero and missing values with average of adjacent nonzero rows"

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

 
    df['RollingTWA_8hr'] = df['converted'].rolling(window=8*60, min_periods=1).mean()
    df['RollingTWA_10hr'] = df['converted'].rolling(window=10*60, min_periods=1).mean()
    df['RollingTWA_12hr'] = df['converted'].rolling(window=12*60, min_periods=1).mean()



    df['RollingMean'] = df['converted'].rolling(window=8*60, min_periods=1).mean()
    df['RollingStd'] = df['converted'].rolling(window=8*60, min_periods=1).std()

    confidence_level = st.radio(
        "Select Confidence Level",
        options=[('95%', 1.96), ('70%', 1.04)],
        format_func=lambda x: x[0]
    )

    z_score = confidence_level[1]

    for hrs, label, limit in [(8, '8hr', twa_8hr_limit), (10, '10hr', twa_10hr_limit), (12, '12hr', twa_12hr_limit)]:
        window = hrs * 60
        df[f'RollingTWA_{label}'] = df['converted'].rolling(window=window, min_periods=1).mean()
        rolling_std = df['converted'].rolling(window=window, min_periods=1).std()
        df[f'UCL_{label}'] = df[f'RollingTWA_{label}'] + z_score * rolling_std
        df[f'LCL_{label}'] = (df[f'RollingTWA_{label}'] - z_score * rolling_std).clip(lower=0)

    if use_stel and stel_limit:
        stel_minutes = int(time_interval.split('-')[0]) * 60
        df['RollingSTEL'] = df['converted'].rolling(window=stel_minutes, min_periods=1).mean()
        std_stel = df['converted'].rolling(window=stel_minutes, min_periods=1).std()
        df['UCL_STEL'] = df['RollingSTEL'] + z_score * std_stel
        df['LCL_STEL'] = (df['RollingSTEL'] - z_score * std_stel).clip(lower=0)


    st.header("Time Series", help ="This graph compares your concentrations to the exposure limits you entered, so you can visualize how much of the day you are over (or under) that limit. ")
    
    # Graph 1
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df.index, y=df['converted'],
        mode='lines',
        name='Converted Data',
        line=dict(color='blue')
    ))
    fig1.update_layout(
        title=f'Time Series of Concentration ({unit_display})',
        xaxis_title='Time',
        yaxis_title=f'Concentration ({unit_display})',
        legend_title='Legend'
    )

    if use_ceiling and ceiling_limit:
        fig1.add_hline(y=ceiling_limit, line_dash="dash", line_color="black", annotation_text="Ceiling Limit")

    st.plotly_chart(fig1)

    # Graph 2
    fig2 = go.Figure()

    
    fig2.add_trace(go.Scatter(x=df.index, y=df['RollingTWA_8hr'], name='8-Hr TWA', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['UCL_8hr'], name='UCL 8-Hr', line=dict(color='blue', dash='dot')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['LCL_8hr'], name='LCL 8-Hr', line=dict(color='blue', dash='dash')))
    fig2.add_hline(y=twa_8hr_limit, line_dash="dash", line_color="blue", annotation_text="8-Hr Limit")

    fig2.add_trace(go.Scatter(x=df.index, y=df['RollingTWA_10hr'], name='10-Hr TWA', line=dict(color='green')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['UCL_10hr'], name='UCL 10-Hr', line=dict(color='green', dash='dot')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['LCL_10hr'], name='LCL 10-Hr', line=dict(color='green', dash='dash')))
    fig2.add_hline(y=twa_10hr_limit, line_dash="dash", line_color="green", annotation_text="10-Hr Limit")

    fig2.add_trace(go.Scatter(x=df.index, y=df['RollingTWA_12hr'], name='12-Hr TWA', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['UCL_12hr'], name='UCL 12-Hr', line=dict(color='red', dash='dot')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['LCL_12hr'], name='LCL 12-Hr', line=dict(color='red', dash='dash')))
    fig2.add_hline(y=twa_12hr_limit, line_dash="dash", line_color="red", annotation_text="12-Hr Limit")

    fig2.update_layout(
        title='Rolling TWAs with Confidence Intervals vs Exposure Limits',
        xaxis_title='Time',
        yaxis_title=f'Concentration ({unit_display})',
        legend_title='Legend'
    )
    st.plotly_chart(fig2)

    if use_stel:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df['RollingSTEL'], name='STEL Moving Avg', line=dict(color='purple')))
        fig3.add_trace(go.Scatter(x=df.index, y=df['UCL_STEL'], name='UCL STEL', line=dict(color='purple', dash='dot')))
        fig3.add_trace(go.Scatter(x=df.index, y=df['LCL_STEL'], name='LCL STEL', line=dict(color='purple', dash='dash')))
        fig3.add_hline(y=stel_limit, line_dash="dash", line_color="purple", annotation_text="STEL Limit")

        fig3.update_layout(
            title='Rolling STEL with Confidence Interval vs Limit',
            xaxis_title='Time',
            yaxis_title=f'Concentration ({unit_display})',
            legend_title='Legend'
        )
        st.plotly_chart(fig3)


    st.divider()

    # Leidel Evaluation
    def evaluate_leidel(twa_col, ucl_col, lcl_col, limit, label):
        ucl = df[ucl_col].dropna().round(2)
        lcl = df[lcl_col].dropna().round(2)

        if (lcl > limit).any():
            return label, "A – Noncompliance Exposure", "Certain Fail", ":thumbsdown:", "red"

        if (ucl < limit).all():
            return label, "C – Compliance Exposure", "Sufficiently Certain", ":thumbsup:", "green"

        return label, "B – Possible Overexposure", "Insufficiently Certain", ":thumbsdown:", "orange"

    results = []
    results.append(evaluate_leidel("RollingTWA_8hr", "UCL_8hr", "LCL_8hr", twa_8hr_limit, "8-hour TWA"))
    results.append(evaluate_leidel("RollingTWA_10hr", "UCL_10hr", "LCL_10hr", twa_10hr_limit, "10-hour TWA"))
    results.append(evaluate_leidel("RollingTWA_12hr", "UCL_12hr", "LCL_12hr", twa_12hr_limit, "12-hour TWA"))

    if use_stel and stel_limit:
        results.append(evaluate_leidel("RollingSTEL", "UCL_STEL", "LCL_STEL", stel_limit, "STEL"))

    if use_ceiling and ceiling_limit:
        if (df['converted'] > ceiling_limit).any():
            results.append(("Ceiling", "A – Ceiling Exceeded", "One or more values exceeded the ceiling limit", ":thumbsdown:", "red"))
        else:
            results.append(("Ceiling", "C – Ceiling Not Exceeded", "All values remained below the ceiling limit", ":thumbsup:", "green"))


    st.subheader("Analysis Outcome", help="Classified using Leidel-style logic based on rolling confidence intervals.")

    for label, decision, certainty, thumbs, color in results:
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(thumbs, unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{label}**")
            st.markdown(f"<span style='color: {color}; font-weight: bold;'>{decision}</span>", unsafe_allow_html=True)
            st.markdown(f"*Certainty:* {certainty}")


    with st.expander("What do A, B, and C decisions mean?"):
        st.markdown("""
    | Condition                                      | Certainty              | Outcome                          | Indicator     |
    |-----------------------------------------------|------------------------|-----------------------------------|---------------|
    | Any Moving TWA > (95% LCL of) Limit Value                | **Certain Fail**       | **A – Noncompliance Exposure**    | 👎 Thumbs Down (Red) |
    | Any Moving Average cannot be classified       | **Insufficiently Certain** | **B – Possible Overexposure** | 👎 Thumbs Down (Orange) |
    | All Moving TWAs &lt; (95% UCL of) Limit Value | **Sufficiently Certain** | **C – Compliance Exposure**    | 👍 Thumbs Up (Green) |
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Log-Normality Test", help="This checks if the data appears log-normally distributed, using three statistical tests.")

    with st.expander("Explanation of Normality Tests"):
        st.markdown("""
        ### Anderson-Darling Result  
        Anderson-Darling is a deterministic test for normality with a number of time-series values approximating a 1-minute data log work shift (around 500).

        ### Shapiro-Wilk Result  
        Shapiro-Wilk is a deterministic test for normality that works better than Anderson-Darling when you have a smaller number of time-series values, such as 1-minute data logs for a 15-minute or 30-minute STEL.

        ### Filliben’s Goodness-of-Fit Result  
        Filliben’s Goodness-of-Fit is the method of determining if the distribution (in this case, within a single data log) is lognormal. This is the same method used by IHDA-AIHA.
        """)

    if st.checkbox('Check if data is log-normally distributed'):
       if (df['converted'] <= 0).any():
           st.error("Data contains non-positive values, which cannot be log-transformed.")
       else:
            log_data = np.log(df['converted'].dropna())

            st.write("### Goodness-of-Fit Test Results")

            # Shapiro Wilk Test
            shapiro_stat, shapiro_p = stats.shapiro(log_data)
            st.write(f"**Shapiro-Wilk Test:** Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
            if shapiro_p > 0.05:
                st.success("Shapiro-Wilk: Data appears normal (fail to reject H₀).")
            else:
                st.warning("Shapiro-Wilk: Data does not appear normal (reject H₀).")

            # Anderson Darling Test
            ad_result = stats.anderson(log_data, dist='norm')
            st.write(f"**Anderson-Darling Test:** Statistic = {ad_result.statistic:.4f}")
            for sl, cv in zip(ad_result.significance_level, ad_result.critical_values):
                st.write(f"  - At {sl:.0f}% significance: Critical Value = {cv:.4f}")
            if ad_result.statistic < ad_result.critical_values[2]:  # 5% level
                st.success("Anderson-Darling: Data appears normal (fail to reject H₀).")
            else:
                st.warning("Anderson-Darling: Data does not appear normal (reject H₀).")

            # Filliben's Test
            sorted_data = np.sort(log_data)
            theoretical_quants = stats.norm.ppf(np.linspace(0.01, 0.99, len(log_data)))
            correlation = np.corrcoef(sorted_data, theoretical_quants[:len(sorted_data)])[0, 1]
            st.write(f"**Filliben’s Correlation Coefficient (r):** {correlation:.4f}")
            if correlation > 0.97:
                st.success("Filliben’s Test: Strong normality (r > 0.97).")
            elif correlation > 0.95:
                st.info("Filliben’s Test: Moderate normality (r > 0.95).")
            else:
                st.warning("Filliben’s Test: Weak normality (r ≤ 0.95).")


st.sidebar.divider()


st.sidebar.header("Real-Time Detection Systems (RTDS) Volunteer Committee")
st.sidebar.markdown(( """[Join Us!](https://www.aiha.org/get-involved/volunteer-groups/real-time-detection-systems-committee)"""))


st.sidebar.markdown(("""[Donate](https://www.aiha.org/get-involved/aih-foundation/aih-foundation-scholarships/real-time-detection-systems-scholarship) today to the RTDS Scholarship!"""))

st.sidebar.divider()
st.sidebar.image("Picture1.png")
