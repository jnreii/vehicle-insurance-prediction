import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Page config
st.set_page_config(page_title="Cross-Sell Predictor", page_icon="🚗", layout="wide")

## Page design
ACCENT = "#6941e8"
ACCENT_DARK = "#5733cf"
INK = "#1e1b4b"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&display=swap');

    /* Canvas and typography*/
    .stApp {{
        background-color: #ffffff;
        color: {INK};
    }}
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp td, .stApp th,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button, .stApp input {{
        font-family: 'DM Sans', sans-serif;
    }}
    /* Keep Streamlit's icon font intact*/
    [data-testid="stIconMaterial"], span[class*="material"] {{
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }}

    /*Hide the default top */
    [data-testid="stHeader"] {{ background: transparent; }}

    /* Pull the content up, the default top padding leaves a large empty gap */
    .block-container, [data-testid="stMainBlockContainer"] {{
        padding-top: 2rem;
    }}

    /* Smaller title */
    .stApp h1 {{
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.4rem;
    }}

    h1, h2, h3, h4, p, label, li {{ color: {INK}; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: #453f70; }}
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {INK}; }}

    /* soft shdow and lift on hover*/
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {{
        background-color: #ffffff;
        border-radius: 14px;
        border: 1px solid #ececf5;
        box-shadow: 0 2px 10px rgba(30, 27, 75, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]):hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(30, 27, 75, 0.10);
        border-color: #ddd9f0;
    }}

    /* Pill-shaped tabs with a soft tint*/
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(105, 65, 232, 0.08);
    }}

    /* Scrollbar in the app's colours*/
    ::-webkit-scrollbar {{ width: 10px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: #c4bcf0; border-radius: 5px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #a89ee8; }}

    /*glow for sidebar input*/
    [data-testid="stSidebar"] input:focus {{
        box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.4);
    }}

    /*indigo sidebar */
    [data-testid="stSidebar"] {{ background-color: {INK}; }}
    [data-testid="stSidebar"] * {{ color: #f2f1fb; }}
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div {{ color: {INK}; }}

    /* Primary button */
    .stButton > button[kind="primary"] {{
        background-color: {ACCENT};
        border-color: {ACCENT};
        color: #ffffff;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(105, 65, 232, 0.3);
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {ACCENT_DARK};
        border-color: {ACCENT_DARK};
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(105, 65, 232, 0.4);
    }}

    /* Slider */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
        background-color: {ACCENT} !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:first-child {{
        background: {ACCENT} !important;
    }}
    [data-testid="stSliderThumbValue"] {{ color: #f2f1fb !important; }}

    /* Tab underline and tab label */
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}

    /* Progress bar fill */
    .stProgress > div > div > div > div {{ background-color: {ACCENT} !important; }}
    [data-testid="stProgress"] div[role="progressbar"] > div {{ background-color: {ACCENT} !important; }}
    </style>
    """,
    unsafe_allow_html=True
)

## Load trained model
## try/except so a missing file gives a proper message, not a traceback
try:
    model = joblib.load("insurance_final_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Make sure 'insurance_final_model.pkl' is in the app folder.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

## Input options
yes_no = ['Yes', 'No']

## Region_Code, Policy_Sales_Channel and Vintage are internal codes. Feature importance show they only make up
#about 4.7% of the model's decisions, so I used median values
DEFAULT_REGION_CODE = '28.0'
DEFAULT_POLICY_CHANNEL = '152.0'
DEFAULT_VINTAGE = 150

## 0.5 cutoff
THRESHOLD = 0.50

##input columns for user
REQUIRED_COLUMNS = ['Age', 'Previously_Insured', 'Vehicle_Damage', 'Annual_Premium']


def prepare_features(df_raw):
##puts rows into columns model was trained on
    df = df_raw.copy()

    df['Region_Code'] = DEFAULT_REGION_CODE
    df['Policy_Sales_Channel'] = DEFAULT_POLICY_CHANNEL
    df['Vintage'] = DEFAULT_VINTAGE

    ## One-hot encoding
    df = pd.get_dummies(df,
                        columns=['Region_Code', 'Vehicle_Damage', 'Policy_Sales_Channel'])

    ##Line the columns up 
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    return df


##Sidebar inputs for the single-customer prediction
with st.sidebar:
    st.header("Customer details")

    age_selected = st.slider("Age", 20, 85, 35)

    vehicle_damage_selected = st.selectbox("Has the vehicle been damaged before?", yes_no)

    previously_insured_selected = st.selectbox("Already has vehicle insurance?", yes_no, index=1)

    annual_premium_selected = st.number_input("Annual health premium ($)",
                                              min_value=2000,
                                              max_value=550000,
                                              value=30000,
                                              step=1000)

    st.write("")
    predict_clicked = st.button("Run prediction", type="primary", use_container_width=True)

## Main page
col_title, col_art = st.columns([3, 1])
with col_title:
    st.title("Vehicle Insurance Cross-Sell Predictor 🚗")
    st.write(
        "Predicts whether an existing health insurance customer is likely to be interested in vehicle insurance, so marketing can focus on the customers most likely to say yes."
    )
with col_art:
    if os.path.exists("cars.png"):
        st.image("cars.png", use_container_width=True)

##tabs for one customer and customer list
tab_single, tab_batch = st.tabs(["Single customer", "Score a customer list"])

##Single customer tab
with tab_single:
    if predict_clicked:
        try:
            df_input = pd.DataFrame({
                'Age': [age_selected],
                'Previously_Insured': [1 if previously_insured_selected == 'Yes' else 0],
                'Vehicle_Damage': [vehicle_damage_selected],
                'Annual_Premium': [annual_premium_selected]
            })

            df_input = prepare_features(df_input)

            ## Get probability, then apply the cutoff
            proba = model.predict_proba(df_input)[0][1]
            prediction = 1 if proba >= THRESHOLD else 0

        except Exception as e:
            st.error(f"Something went wrong while making the prediction: {e}")
            st.stop()

        with st.container(border=True):
            if prediction == 1:
                st.success("**Likely interested.** Include this customer in the marketing campaign.")
            else:
                st.warning("**Unlikely to be interested.** Leave this customer out of the campaign.")

            col1, col2= st.columns(2)
            col1.metric("Chance of interest", f"{proba:.0%}")
            col2.metric("Contact cutoff", f"{THRESHOLD:.0%}",
                        help="Anyone at {THRESHOLD:.0%} or higher can be contacted for campaign.")

            ##label for progress bar
            st.progress(float(proba),
                        text=f"Interest level: {proba:.0%} — customers at {THRESHOLD:.0%} or above are contacted")

        ## Explaination for results
        with st.container(border=True):
            st.markdown("**Why this prediction**")

            reasons = []
            if vehicle_damage_selected == 'Yes':
                reasons.append(("Vehicle damaged before", "Yes", "▲ Raises interest"))
            else:
                reasons.append(("Vehicle damaged before", "No", "▼ Lowers interest"))

            if previously_insured_selected == 'Yes':
                reasons.append(("Already has vehicle insurance", "Yes", "▼ Lowers interest, they are already covered"))
            else:
                reasons.append(("Already has vehicle insurance", "No", "▲ Raises interest, they have no cover yet"))

            if 30 <= age_selected <= 60:
                reasons.append(("Age", str(age_selected), "▲ In the age range that responds most"))
            else:
                reasons.append(("Age", str(age_selected), "▼ Outside the age range that responds most"))

            df_reasons = pd.DataFrame(reasons, columns=["Factor", "This customer", "Impact on interest"])
            st.dataframe(df_reasons, hide_index=True, use_container_width=True)

            st.caption(
                "Whether the vehicle has been damaged is by far the biggest factor, followed by age and if the customer already has cover."
            )

        ##Run the model again with one detail flipped each time, to show how much each factor is actually doing
        with st.container(border=True):
            st.markdown("**What would change this?**")

            def rescore(damage, insured):
                df_alt = pd.DataFrame({
                    'Age': [age_selected],
                    'Previously_Insured': [1 if insured == 'Yes' else 0],
                    'Vehicle_Damage': [damage],
                    'Annual_Premium': [annual_premium_selected]
                })
                return model.predict_proba(prepare_features(df_alt))[0][1]

            ##flip the two biggest factors
            flipped_damage = 'No' if vehicle_damage_selected == 'Yes' else 'Yes'
            flipped_insured = 'No' if previously_insured_selected == 'Yes' else 'Yes'

            proba_damage_flipped = rescore(flipped_damage, previously_insured_selected)
            proba_insured_flipped = rescore(vehicle_damage_selected, flipped_insured)

            df_whatif = pd.DataFrame({
                "If instead...": [
                    f"the vehicle had {'no damage' if flipped_damage == 'No' else 'been damaged'}",
                    f"the customer {'had no' if flipped_insured == 'No' else 'already had'} vehicle insurance"
                ],
                "Chance of interest": [f"{proba_damage_flipped:.0%}", f"{proba_insured_flipped:.0%}"],
                "Change": [
                    f"{(proba_damage_flipped - proba):+.0%}",
                    f"{(proba_insured_flipped - proba):+.0%}"
                ]
            })
            st.dataframe(df_whatif, hide_index=True, use_container_width=True)

        st.caption(
            "This prediction is a guide for prioritising customers to contact, not a guarantee of interest. Final decisions should be made by the marketing team."
        )

    else:
        st.info("Fill in the customer's details on the left, then click Run prediction.")

##customer list tab
with tab_batch:

    st.write(
        "Upload a customer list and get every row scored at once, sorted so the strongest prospects are at the top."
    )

    ##Example file
    df_template = pd.DataFrame({
        'Age': [45, 23, 51],
        'Previously_Insured': [0, 1, 0],
        'Vehicle_Damage': ['Yes', 'No', 'Yes'],
        'Annual_Premium': [35000, 28000, 42000]
    })

    st.download_button(
        "Download an example file! (CSV)",
        data=df_template.to_csv(index=False),
        file_name="customer_list_example.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload your customer list (CSV)", type="csv")

    if uploaded_file is not None:
        ##error handling
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception:
            st.error("That file could not be read as a CSV. Please upload a valid CSV file.")
            st.stop()

        ##empty file
        if df_upload.empty:
            st.error("That file has no rows in it. Please upload a file with at least one customer.")
            st.stop()

        ##missing columns
        col_missing = [c for c in REQUIRED_COLUMNS if c not in df_upload.columns]
        if col_missing:
            st.error(
                f"These required columns are missing: {', '.join(col_missing)}. "
                "Download the example file above to see the correct format."
            )
            st.stop()

        ##check if the values make sense
        df_clean = df_upload[REQUIRED_COLUMNS].copy()

        ## anything not a number turns into NaN
        df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
        df_clean['Annual_Premium'] = pd.to_numeric(df_clean['Annual_Premium'], errors='coerce')
        df_clean['Previously_Insured'] = pd.to_numeric(df_clean['Previously_Insured'], errors='coerce')

        if df_clean[['Age', 'Annual_Premium', 'Previously_Insured']].isna().any().any():
            st.error(
                "Age, Annual_Premium and Previously_Insured must all be numbers. "
                "Some rows contain text or blanks."
            )
            st.stop()

        ## has to be 0 or 1
        if not df_clean['Previously_Insured'].isin([0, 1]).all():
            st.error("Previously_Insured must be 0 (no) or 1 (yes) in every row.")
            st.stop()

        ## has to be Yes or No
        df_clean['Vehicle_Damage'] = df_clean['Vehicle_Damage'].astype(str).str.strip().str.capitalize()
        if not df_clean['Vehicle_Damage'].isin(['Yes', 'No']).all():
            st.error("Vehicle_Damage must be 'Yes' or 'No' in every row.")
            st.stop()

        ##score everything
        try:
            X_upload = prepare_features(df_clean)
            proba_all = model.predict_proba(X_upload)[:, 1]
        except Exception as e:
            st.error(f"Something went wrong while scoring the list: {e}")
            st.stop()

        ##Puts results into the original rows
        df_results = df_upload.copy()
        df_results['Chance of interest (%)']= (proba_all * 100).round(0).astype(int)
        df_results['Recommend contacting?'] = np.where(proba_all >= THRESHOLD, 'Yes', 'No')

        ## Best prospects at the top
        df_results = df_results.sort_values('Chance of interest (%)', ascending=False)

        n_contact = int((proba_all >= THRESHOLD).sum())
        st.success(f"Scored {len(df_results)} customers. {n_contact} are recommended for contact.")

        st.dataframe(df_results, hide_index=True, use_container_width=True)

        st.download_button(
            "Download the scored list (CSV)",
            data=df_results.to_csv(index=False),
            file_name="scored_customer_list.csv",
            mime="text/csv"
        )
