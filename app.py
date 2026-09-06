
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Smart Car Valuation",
    page_icon="🚗",
    layout="centered"
)


# ================================================================
# LOAD DATASET (for dropdown options and default values)
# ================================================================

DATASET_FILE = "cleaned_second_hand_car_dataset.xlsx"

try:
    df = pd.read_excel(DATASET_FILE)
except FileNotFoundError:
    st.error(f"Error: Dataset file not found at {DATASET_FILE}. Please ensure the file exists.")
    st.stop()

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_", regex=False)
)


# ================================================================
# LOAD MODEL AND PREPROCESSOR
# ================================================================

MODEL_FILE = "/content/random_forest_regression_model.pkl"
PREPROCESSOR_FILE = "/content/car_price_preprocessor.pkl"

try:
    rf_model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
except FileNotFoundError:
    st.error(f"Error: Model or preprocessor file not found. Ensure {MODEL_FILE} and {PREPROCESSOR_FILE} exist.")
    st.stop()

# ================================================================
# EXPECTED MODEL FEATURES (from notebook context)
# ================================================================

MODEL_COLUMNS = [
    "car_name", "brand", "model", "vehicle_age", "km_driven",
    "seller_type", "fuel_type", "transmission_type", "mileage",
    "engine", "max_power", "seater"
]

# ================================================================
# PREMIUM / LUXURY THRESHOLD (from notebook context)
# ================================================================

PREMIUM_THRESHOLD = 1000000


# ================================================================
# STREAMLIT UI
# ================================================================

st.title("Smart Car Valuation")
st.write("Second Hand Car Price Estimation Based on Multiple Factors and Advanced Models.")

st.markdown("--- --- --- --- --- --- --- --- --- ---")

# Define callback functions to reset dependent dropdowns
def on_car_name_change_callback():
    # Reset selected brand and model when car name changes
    st.session_state['selected_brand_key'] = None
    st.session_state['selected_model_key'] = None

def on_brand_change_callback():
    # Reset selected model when brand changes
    st.session_state['selected_model_key'] = None

# --- Car Name (main selector) ---
car_name_options = sorted(df['car_name'].unique())
# Initialize 'car_name_select_key' in session_state if it doesn't exist
if 'car_name_select_key' not in st.session_state:
    st.session_state.car_name_select_key = car_name_options[0]

car_name_selected = st.selectbox(
    "Car Name",
    car_name_options,
    key="car_name_select_key", # This key links to st.session_state.car_name_select_key
    on_change=on_car_name_change_callback # Call reset function on change
)

# --- Brand Logic ---
filtered_df_for_brand = df[df['car_name'] == car_name_selected]
brand_options_for_car = sorted(filtered_df_for_brand['brand'].unique())

# Ensure there's always at least one option to prevent selectbox errors
if not brand_options_for_car:
    brand_options_for_car = ["No Brand Available"]

# Determine the value that should be selected for 'Brand'
current_brand_value_in_session = st.session_state.get('selected_brand_key')

# If the session state value for brand is None, or not in the current options,
# then explicitly set it to the first option of the new list.
if current_brand_value_in_session is None or current_brand_value_in_session not in brand_options_for_car:
    st.session_state.selected_brand_key = brand_options_for_car[0]

# Now, retrieve the actual index of the value that IS in session_state
# This ensures the selectbox's 'index' parameter correctly points to the desired default.
brand_display_index = brand_options_for_car.index(st.session_state.selected_brand_key)

brand_selected = st.selectbox(
    "Brand",
    brand_options_for_car,
    index=brand_display_index,
    key="selected_brand_key", # Value is directly linked to this session state key
    on_change=on_brand_change_callback # Call reset function on change
)

# --- Model Logic ---
filtered_df_for_model = df[(df['car_name'] == car_name_selected) & (df['brand'] == brand_selected)]
model_options_for_brand = sorted(filtered_df_for_model['model'].unique())

# Ensure there's always at least one option to prevent selectbox errors
if not model_options_for_brand:
    model_options_for_brand = ["No Model Available"]

# Determine the value that should be selected for 'Model'
current_model_value_in_session = st.session_state.get('selected_model_key')

# If the session state value for model is None, or not in the current options,
# then explicitly set it to the first option of the new list.
if current_model_value_in_session is None or current_model_value_in_session not in model_options_for_brand:
    st.session_state.selected_model_key = model_options_for_brand[0]

# Now, retrieve the actual index of the value that IS in session_state
model_display_index = model_options_for_brand.index(st.session_state.selected_model_key)

model_selected = st.selectbox(
    "Model",
    model_options_for_brand, # Corrected to model_options_for_brand
    index=model_display_index,
    key="selected_model_key" # Value is directly linked to this session state key
)

# Assign to original variable names for downstream use (for compatibility with existing model_columns etc.)
car_name = car_name_selected
brand = brand_selected
model = model_selected

with st.form("prediction_form"):
    st.subheader("Enter Vehicle Details (Other Features)")

    col1, col2 = st.columns(2)

    # Other dropdowns (Fuel Type, Transmission, Seller Type, Seater) - independent for now
    fuel_type_options = sorted(df['fuel_type'].unique())
    fuel_type = col2.selectbox("Fuel Type", fuel_type_options, key="fuel_type_select")

    transmission_type_options = sorted(df['transmission_type'].unique())
    transmission_type = col1.selectbox("Transmission Type", transmission_type_options, key="transmission_type_select")

    seller_type_options = sorted(df['seller_type'].unique())
    seller_type = col2.selectbox("Seller Type", seller_type_options, key="seller_type_select")

    seater_options = sorted(df['seater'].unique())
    seater = col1.selectbox("Seater", seater_options, key="seater_select")

    # Numerical inputs
    st.markdown("### Numerical Features")
    col3, col4 = st.columns(2)
    vehicle_age = col3.number_input("Vehicle Age (years)", min_value=1, max_value=25, value=5, key="vehicle_age_input")
    km_driven = col4.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000, key="km_driven_input")
    mileage = col3.number_input("Mileage (km/l)", min_value=5.0, max_value=50.0, value=15.0, step=0.1, key="mileage_input")
    engine = col4.number_input("Engine (cc)", min_value=500, max_value=5000, value=1200, step=10, key="engine_input")
    max_power = col3.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, value=80.0, step=0.1, key="max_power_input")

    submitted = st.form_submit_button("Estimate Selling Price")

    if submitted:
        # Prepare input for prediction
        input_data = {
            "car_name": car_name,
            "brand": brand,
            "model": model,
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seater": seater
        }

        input_df = pd.DataFrame([input_data])

        # Ensure the order of columns matches MODEL_COLUMNS if needed by preprocessor
        input_df = input_df[MODEL_COLUMNS]

        # Transform input using the preprocessor
        transformed_input = preprocessor.transform(input_df)

        # Make prediction with the Random Forest model
        prediction = rf_model.predict(transformed_input)
        predicted_price = float(np.asarray(prediction).reshape(-1)[0])
        predicted_price = max(0, predicted_price) # Ensure price is not negative

        # Determine category
        category = "Premium / Luxury" if predicted_price >= PREMIUM_THRESHOLD else "Standard"

        st.subheader("Predicted Selling Price")
        st.markdown(f"""
        <div style="background-color:#007bff; padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h2 style="margin:0;">₹ {predicted_price:,.0f}</h2>
            <p style="margin:5px 0 0;">Category: <strong>{category}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.info("Disclaimer: This is an estimated price based on the input data and selected vehicle features.")

st.markdown("--- --- --- --- --- --- --- --- --- ---")
st.caption("Smart Car Valuation System | Developed by Tanish Malvankar")
