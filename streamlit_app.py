import streamlit as st
from autogluon.tabular import TabularPredictor
import pandas as pd
import dill
import plotly.graph_objects as go

# Enable wide mode
st.set_page_config(layout="wide")

def create_ring_plot(probability, title):
    """
    Creates a ring (donut) plot to visualize risk probability.

    Parameters:
    - probability (float): Risk probability between 0 and 1.
    - title (str): Title of the plot.

    Returns:
    - Plotly figure object.
    """
    # Determine color based on risk level
    if probability < 0.33:
        color = 'green'
    elif probability < 0.66:
        color = 'orange'
    else:
        color = 'red'
    
    fig = go.Figure(go.Pie(
        values=[probability, 1 - probability],
        hole=0.7,
        marker=dict(colors=[color, 'lightgray']),
        hoverinfo='none'
    ))

    fig.update_traces(textinfo='none')
    fig.update_layout(
        annotations=[
            dict(text=f"{probability:.2%}", x=0.5, y=0.5, font_size=20, showarrow=False)
        ],
        title_text=title,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Load session data
import pickle
from pathlib import Path
import pathlib
pathlib.WindowsPath = pathlib.PosixPath

def convert_strings_to_paths(obj):
    if isinstance(obj, str) and ('path' in obj.lower() or '/' in obj):
        return Path(obj)
    elif isinstance(obj, dict):
        return {k: convert_strings_to_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_strings_to_paths(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_strings_to_paths(item) for item in obj)
    # Add more conditions if necessary
    return obj

def load_pickle_file(filename):
    """Utility function to load a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def loadsupport():
    # Load predictorall from predictorall.pkl
    with open('predictorall.pkl', 'rb') as f:
        predictorall = pickle.load(f)
    predictorall = convert_strings_to_paths(predictorall)
    
    # Load predictorcardfull from predictorcard.pkl
    with open('predictorcard.pkl', 'rb') as f:
        predictorcardfull = pickle.load(f)
    predictorcardfull = convert_strings_to_paths(predictorcardfull)
    
    # Load predictorsepsisfull from predictorsepsis.pkl
    with open('predictorsepsis.pkl', 'rb') as f:
        predictorsepsisfull = pickle.load(f)
    predictorsepsisfull = convert_strings_to_paths(predictorsepsisfull)
    
    # Load emer dataframe from emer.pkl
    with open('emer.pkl', 'rb') as f:
        emer = pickle.load(f)
    emer = convert_strings_to_paths(emer)  # Apply only if necessary for the dataframe
    
    return predictorall, predictorcardfull, predictorsepsisfull, emer

predictor, card_predictor, sepsis_predictor, emer = loadsupport()

# Load the DataFrame with average values
# Assuming 'emer' DataFrame is loaded via dill.load_session()
# Drop rows with any missing values to clean the data
emer_clean = emer.dropna(how='any', inplace=False)

# Get features and types from all three predictors
allcause_features = set(predictor.feature_metadata.get_features())
card_features = set(card_predictor.feature_metadata.get_features())
sepsis_features = set(sepsis_predictor.feature_metadata.get_features())

# Combine all features
all_features = allcause_features.union(card_features).union(sepsis_features)
feature_names_list = list(all_features)
num_features = len(feature_names_list)

# Get feature types
feature_types = {}
feature_types.update(predictor.feature_metadata.type_map_raw)
feature_types.update(card_predictor.feature_metadata.type_map_raw)
feature_types.update(sepsis_predictor.feature_metadata.type_map_raw)

# Set up the Streamlit app
st.title('Mortality Risk Prediction for Hemodialysis Patients in Intensive Care Units')
st.write('Enter the following values for a hemodialysis patient in the Intensive Care Unit to predict their death risk during hospitalization:')

# Display names mapping
display_names = {
    '昏迷/意识丧失模糊': 'Coma(0/1)',
    '心肺复苏': 'Use of Cardiopulmonary Resuscitation(0/1)',
    '脓毒症': 'Sepsis(0/1)',
    '舒张压': 'Diastolic Blood Pressure(mmHg)',
    '高敏肌钙蛋白T(TnT-T)(发光法)': 'Troponin T(ng/ml)',
    '肌红蛋白(肌血球素)MYO': 'Myoglobin(ng/ml)',
    '中性分叶粒细胞NEUT#': 'Neutrophil Count(×10^9/L)',
    '凝血酶原活动度PT(%)': 'Prothrombin Time Activity(%)',
    '大血小板百分率': 'Platelet Large Cell Ratio(%)',
    '活化部分凝血活酶时间APTT': 'Activated Partial Thromboplastin Time(s)',
    'NAR': 'NAR',
    '收缩压': 'Systolic Blood Pressure(mmHg)',
    '肌酸激酶同工酶CK-MB(mass法)': 'Creatine Kinase-MB(ng/ml)',
    '阴离子间隙AG': 'Anion Gap(mmol/L)',
    'C反应蛋白': 'C-reactive protein(mg/L)',
    '平均血小板体积MPV': 'Mean Platelet Volume(fL)',
    '总胆红素TBIL( µmol/L)': 'Total bilirubin(µmol/L)',
    'PIV': 'PIV',
    'SIRI': 'SIRI',
    'CAR': 'CAR',
    '中性分叶粒细胞NEUT%': 'Neutrophil Percentage(%)',
    'B型钠尿肽前体(NT-proBNP)': 'NT-proBNP(pg/mL)',
    'NLR': 'NLR',
    '嗜酸细胞EO%': 'Eosinophil Percentage(%)',
    '平均红细胞血红蛋白浓度MCHC': 'Mean Corpuscular Hemoglobin Concentration(g/L)',
    '心衰': 'Heart Failure(0/1)'
}

# Ensure all features have a display name
for feature in feature_names_list:
    if feature not in display_names:
        display_names[feature] = feature  # Use the feature name itself if no display name is provided

# Bulk input option with a specific order and variable definitions
st.write('Alternatively, you can input all values separated by commas. If entered this way, the values will automatically populate each input field to ensure accurate recognition.')

# Define the desired order for bulk input
bulk_input_order = [
    'Mean Corpuscular Hemoglobin Concentration(g/L)',
    'Creatine Kinase-MB(ng/ml)',
    'Neutrophil Percentage(%)',
    'Systolic Blood Pressure(mmHg)',
    'Troponin T(ng/ml)',
    'NAR',
    'Sepsis(0/1)',
    'NLR',
    'Platelet Large Cell Ratio(%)',
    'Coma(0/1)',
    'C-reactive protein(mg/L)',
    'PIV',
    'Neutrophil Count(×10^9/L)',
    'SIRI',
    'Total bilirubin(µmol/L)',
    'Prothrombin Time Activity(%)',
    'Diastolic Blood Pressure(mmHg)',
    'Mean Platelet Volume(fL)',
    'Myoglobin(ng/ml)',
    'Anion Gap(mmol/L)',
    'NT-proBNP(pg/mL)',
    'Use of Cardiopulmonary Resuscitation(0/1)',
    'CAR',
    'Eosinophil Percentage(%)',
    'Activated Partial Thromboplastin Time(s)',
    'Heart Failure(0/1)'
]

# Map display names to feature names
display_name_to_feature = {v: k for k, v in display_names.items()}

# Create an ordered list of feature names based on bulk input order
feature_names_list_ordered = []
for name in bulk_input_order:
    feature = display_name_to_feature.get(name)
    if feature:
        feature_names_list_ordered.append(feature)
    else:
        st.error(f"Display name '{name}' does not correspond to any feature.")
        st.stop()

# Update number of features
num_features = len(feature_names_list_ordered)

# Display the ordered list of variable names
st.write(', '.join(bulk_input_order))

# Add definitions for some variables
st.write('Definition of some variables:')
st.write('(1) PIV: (Neutrophil Count * Platelet Count * Monocyte Count) / Lymphocyte Count')
st.write('(2) SIRI: (Neutrophil Count * Monocyte Count) / Lymphocyte Count')
st.write('(3) NAR: Neutrophil Count / Albumin')
st.write('(4) NLR: Neutrophil Count / Lymphocyte Count')
st.write('(5) CAR: C-Reactive Protein / Albumin')

bulk_input = st.text_area('Enter all values separated by commas. If entered this way, the values will automatically populate each input field to ensure accurate recognition.', value='', height=100)

# Dictionary to store user inputs
input_data = {}
missing_features = []

# Arrange input fields into columns
columns_per_row = 6
rows = (num_features + columns_per_row - 1) // columns_per_row  # Ceiling division

# Process bulk input if provided
if bulk_input.strip() != '':
    bulk_values = bulk_input.strip().split(',')
    if len(bulk_values) != len(feature_names_list_ordered):
        st.error('The number of values entered does not match the number of features.')
    else:
        for feature, value in zip(feature_names_list_ordered, bulk_values):
            feature_type = feature_types[feature]
            value = value.strip()
            if feature_type in ['int', 'float']:
                try:
                    value = float(value)
                except ValueError:
                    st.error(f"Invalid numeric value for {display_names.get(feature, feature)}: {value}")
                    st.stop()
            elif feature_type == 'datetime':
                try:
                    value = pd.to_datetime(value)
                except ValueError:
                    st.error(f"Invalid date format for {display_names.get(feature, feature)}: {value}")
                    st.stop()
            input_data[feature] = value

# Create input fields dynamically
for row in range(rows):
    cols = st.columns(columns_per_row)
    for idx in range(columns_per_row):
        feature_idx = row * columns_per_row + idx
        if feature_idx < num_features:
            feature = feature_names_list_ordered[feature_idx]
            feature_type = feature_types[feature]
            display_name = display_names.get(feature, feature)

            with cols[idx]:
                default_value = input_data.get(feature, '')
                if default_value == '':
                    if feature_type in ['int', 'float']:
                        value = st.number_input(f"{display_name}:", key=feature)
                    elif feature_type == 'object':
                        value = st.text_input(f"{display_name}:", key=feature)
                    elif feature_type == 'datetime':
                        value = st.date_input(f"{display_name}:", key=feature)
                    else:
                        value = st.text_input(f"{display_name}:", key=feature)
                else:
                    if feature_type in ['int', 'float']:
                        value = st.number_input(f"{display_name}:", value=float(default_value), key=feature)
                    elif feature_type == 'object':
                        value = st.text_input(f"{display_name}:", value=str(default_value), key=feature)
                    elif feature_type == 'datetime':
                        value = st.date_input(f"{display_name}:", value=pd.to_datetime(default_value), key=feature)
                    else:
                        value = st.text_input(f"{display_name}:", value=str(default_value), key=feature)
                input_data[feature] = value

# When 'Predict' button is clicked
if st.button('Predict'):
    # Create DataFrame from input_data
    input_df = pd.DataFrame([input_data])

    # Handle missing values and keep track of filled values
    missing_features = []
    missing_values_used = {}

    for feature in feature_names_list_ordered:
        if pd.isnull(input_df.loc[0, feature]) or input_df.loc[0, feature] == '':
            feature_type = feature_types[feature]
            if feature_type in ['int', 'float']:
                # Fill with mean
                mean_value = emer_clean[feature].mean()
                input_df.loc[0, feature] = mean_value
                missing_features.append(display_names.get(feature, feature))
                missing_values_used[display_names.get(feature, feature)] = mean_value
            elif feature_type == 'object':
                # Fill with mode
                mode_value = emer_clean[feature].mode()[0]
                input_df.loc[0, feature] = mode_value
                missing_features.append(display_names.get(feature, feature))
                missing_values_used[display_names.get(feature, feature)] = mode_value
            else:
                # Handle other types if necessary
                input_df.loc[0, feature] = None
                missing_features.append(display_names.get(feature, feature))
                missing_values_used[display_names.get(feature, feature)] = None

    # Handle datetime features
    datetime_features = [feature for feature, ftype in feature_types.items() if ftype == 'datetime']
    for feature in datetime_features:
        input_df[feature] = pd.to_datetime(input_df[feature])

    # Make predictions
    prediction = predictor.predict(input_df)
    probability = predictor.predict_proba(input_df)
    card_prediction = card_predictor.predict(input_df) 
    card_probability = card_predictor.predict_proba(input_df)
    sepsis_prediction = sepsis_predictor.predict(input_df)
    sepsis_probability = sepsis_predictor.predict_proba(input_df)

    # Extract probabilities
    risk_of_death_probability = probability.iloc[0][1]
    card_risk_probability = card_probability.iloc[0][1]
    sepsis_risk_probability = sepsis_probability.iloc[0][1]

    # Display results in parallel
    st.subheader('Prediction Results')

    cols = st.columns(3)

    with cols[0]:
        st.markdown("### All-cause Mortality")
        if prediction.iloc[0] == 0:
            prediction_text = 'Low risk of mortality'
        else:
            prediction_text = 'Elevated mortality risk, requiring intervention'
        st.write(f"**Prediction:** {prediction_text}")
        #st.progress(risk_of_death_probability)
        fig = create_ring_plot(risk_of_death_probability, "Probability of Mortality")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Predicted Risk of mortality: {risk_of_death_probability:.2%}")

    with cols[1]:
        st.markdown("### Cardiovascular Mortality")
        if card_prediction.iloc[0] == 0:
            card_prediction_text = 'Low risk of cardiovascular death'
        else:
            card_prediction_text = 'Elevated cardiovascular mortality risk, requiring intervention'
        st.write(f"**Prediction:** {card_prediction_text}")
        #st.progress(card_risk_probability)
        fig = create_ring_plot(card_risk_probability, "Probability of Cardiovascular Mortality")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Predicted Risk of Cardiovascular mortality: {card_risk_probability:.2%}")

    with cols[2]:
        st.markdown("### Infection-related Mortality")
        if sepsis_prediction.iloc[0] == 0:
            sepsis_prediction_text = 'Low risk of infection-related mortality'
        else:
            sepsis_prediction_text = 'Elevated infection-related mortality risk, requiring intervention'
        st.write(f"**Prediction:** {sepsis_prediction_text}")
        #st.progress(sepsis_risk_probability)
        fig = create_ring_plot(sepsis_risk_probability, "Probability of Infection-related Mortality")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Predicted Risk of Infection-related mortality: {sepsis_risk_probability:.2%}")

    # Output missing features filled with averages and the values used
    if missing_features:
        st.warning("The following variables were missing and have been filled with average values:")
        for var in missing_features:
            value_used = missing_values_used[var]
            st.write(f"{var}: {value_used}")
