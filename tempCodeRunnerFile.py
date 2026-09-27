import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(
    page_title="Diabetes AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS with Healthcare Background ---
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main container with healthcare background */
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Background with diabetes/medical theme - heavily blurred */
    .stApp {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(240, 248, 255, 0.92) 100%),
                    url('https://images.unsplash.com/photo-1631549916768-4119b2e5f926?w=1920&q=80') center/cover fixed;
        background-blend-mode: overlay;
    }
    
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('https://images.unsplash.com/photo-1631549916768-4119b2e5f926?w=1920&q=80') center/cover fixed;
        filter: blur(8px);
        opacity: 0.15;
        z-index: -1;
        pointer-events: none;
    }
    
    /* Diabetes-themed subtle pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(66, 153, 225, 0.04) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(72, 187, 120, 0.04) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
        border-right: 2px solid #e2e8f0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* Card-like containers for metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
    }
    
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #4a5568;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: #4a5568;
        font-weight: 600;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Headers with gradient */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        padding: 20px 0;
    }
    
    h2, h3 {
        color: #2d3748;
        font-weight: 600;
    }
    
    /* Custom success/error boxes with glassmorphism */
    .stAlert {
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, rgba(66, 153, 225, 0.1) 0%, rgba(49, 130, 206, 0.1) 100%);
        border-left: 4px solid #3182ce;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 161, 105, 0.1) 100%);
        border-left: 4px solid #38a169;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.1) 0%, rgba(229, 62, 62, 0.1) 100%);
        border-left: 4px solid #e53e3e;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(237, 137, 54, 0.1) 0%, rgba(221, 107, 32, 0.1) 100%);
        border-left: 4px solid #dd6b20;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    
    /* Form containers */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        backdrop-filter: blur(10px);
    }
    
    /* Slider styling with darker, more visible labels */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Make slider labels MUCH darker and more visible */
    .stSlider label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    /* Slider values */
    .stSlider [data-baseweb="slider"] {
        color: #2d3748 !important;
    }
    
    /* All form labels darker */
    label, .stMarkdown label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    /* Subheaders in sidebar darker */
    .stForm h3, .stForm h2 {
        color: #1a202c !important;
        font-weight: 700 !important;
    }
    
    /* Number input styling with darker labels */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        color: #1a202c;
        font-weight: 600;
    }
    
    .stNumberInput label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    /* Medical icons decoration */
    .decoration-icon {
        position: fixed;
        opacity: 0.03;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Plotly chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler_svm.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model_and_scaler()

# --- Sidebar Inputs ---
st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <img src='https://cdn-icons-png.flaticon.com/512/2966/2966327.png' width='100' style='filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));'/>
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("🏥 Patient Data")
st.sidebar.markdown("Enter the clinical details below.")

with st.sidebar.form("prediction_form"):
    st.subheader("👤 Demographics")
    age = st.slider('Age', 21, 100, 30, help="Patient's age in years")
    pregnancies = st.number_input('Pregnancies', 0, 20, 0, help="Number of times pregnant")

    st.subheader("🩺 Vitals & Labs")
    glucose = st.slider('Glucose (mg/dL)', 0, 200, 120, help="Plasma glucose concentration")
    bp = st.slider('Blood Pressure (mm Hg)', 0, 130, 70, help="Diastolic blood pressure")
    skin = st.slider('Skin Thickness (mm)', 0, 100, 20, help="Triceps skin fold thickness")
    insulin = st.slider('Insulin (mu U/ml)', 0, 900, 80, help="2-Hour serum insulin")
    
    col_bmi, col_dpf = st.columns(2)
    with col_bmi:
        bmi = st.number_input('BMI', 10.0, 70.0, 25.0, 0.1, help="Body Mass Index")
    with col_dpf:
        dpf = st.number_input('DPF', 0.0, 2.5, 0.5, 0.01, help="Diabetes Pedigree Function (Genetic Score)")
    
    st.markdown("---")
    predict_btn = st.form_submit_button("🔮 Analyze Risk", type="primary", use_container_width=True)

# --- Main Page Content ---
if model is None or scaler is None:
    st.error("❌ **System Error**: Model files not found.")
    st.code("python diabetes_prediction.py", language="bash")
    st.stop()

if not predict_btn:
    # Initial "Empty State" Dashboard
    st.title("🩺 Diabetes Risk AI")
    st.markdown("### Welcome to the Clinical Decision Support System")
    
    # Hero section with background
    st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%), 
                    url("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=1200&q=80") center/cover; 
                    padding: 40px; border-radius: 15px; margin: 20px 0;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.3);'>
            <h2 style='color: #1a202c; margin: 0; text-shadow: 0 2px 4px rgba(255,255,255,0.8);'>🎯 AI-Powered Diabetes Risk Assessment</h2>
            <p style='color: #2d3748; font-size: 1.1em; margin-top: 10px; text-shadow: 0 1px 2px rgba(255,255,255,0.8);'>
                Leveraging machine learning to predict diabetes risk with clinical precision
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.9); padding: 25px; border-radius: 12px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.05); height: 280px;'>
                <h3 style='color: #667eea;'>📊 How It Works</h3>
                <ol style='color: #4a5568; line-height: 1.8;'>
                    <li>Enter clinical parameters in the sidebar</li>
                    <li>AI analyzes patterns from 768+ records</li>
                    <li>Receive probability score & insights</li>
                    <li>Get personalized care recommendations</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.9); padding: 25px; border-radius: 12px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.05); height: 280px;'>
                <h3 style='color: #764ba2;'>🔬 Model Performance</h3>
                <div style='text-align: center; margin-top: 30px;'>
                    <div style='font-size: 3em; font-weight: bold; color: #667eea;'>78%</div>
                    <div style='color: #4a5568; margin-top: 10px;'>Accuracy Score</div>
                    <div style='color: #718096; font-size: 0.9em; margin-top: 15px;'>Support Vector Machine (SVM)</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.9); padding: 25px; border-radius: 12px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.05); height: 280px;'>
                <h3 style='color: #48bb78;'>✨ Key Features</h3>
                <ul style='color: #4a5568; line-height: 1.8;'>
                    <li>Real-time risk prediction</li>
                    <li>Interactive visualizations</li>
                    <li>Personalized care plans</li>
                    <li>Dietary & lifestyle guidance</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Start by entering patient data in the sidebar to generate a comprehensive risk assessment.**")

else:
    # --- Prediction Logic ---
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_std = scaler.transform(input_data)
    prediction = model.predict(input_std)[0]
    
    try:
        probability = model.predict_proba(input_std)[0]  
        prob_positive = probability[1] * 100
    except:
        prob_positive = 100 if prediction == 1 else 0

    # Determine Risk Category
    risk_category = "High" if prob_positive > 50 else "Low"

    # --- Results Dashboard ---
    st.title("📊 Analysis Results")
    
    # Top-level Status Banner
    if risk_category == "High":
        st.error(f"### 🔴 High Risk Detected ({prob_positive:.1f}%)")
    else:
        st.success(f"### 🟢 Low Risk Detected ({prob_positive:.1f}%)")

    # Tabs for organized view
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧬 Vitals Analysis", "🍎 Lifestyle & Diet"])

    with tab1:
        col_gauge, col_metrics = st.columns([1, 1])
        
        with col_gauge:
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_positive,
                title={'text': "Diabetic Probability", 'font': {'size': 24, 'color': '#2d3748'}},
                number={'suffix': "%", 'font': {'size': 40, 'color': '#2d3748'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#4a5568"},
                    'bar': {'color': "#667eea", 'thickness': 0.75},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(72, 187, 120, 0.3)"},
                        {'range': [40, 70], 'color': "rgba(237, 137, 54, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(245, 101, 101, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "#e53e3e", 'width': 4}, 
                        'thickness': 0.75, 
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                height=350, 
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(255, 255, 255, 0.9)',
                font={'family': 'Poppins, sans-serif'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_metrics:
            st.subheader("🎯 Key Drivers")
            m1, m2 = st.columns(2)
            m1.metric("Glucose", f"{glucose} mg/dL", delta="High" if glucose > 140 else "Normal", delta_color="inverse")
            m2.metric("BMI", f"{bmi}", delta="High" if bmi > 25 else "Normal", delta_color="inverse")
            
            m3, m4 = st.columns(2)
            m3.metric("Age", f"{age} yrs")
            m4.metric("Genetic Score", f"{dpf:.2f}")
            
            # Additional context
            st.markdown("<br>", unsafe_allow_html=True)
            if glucose > 140:
                st.warning("⚠️ Elevated glucose levels detected")
            if bmi > 30:
                st.warning("⚠️ BMI indicates obesity range")

    with tab2:
        st.subheader("📊 Patient vs. Healthy Average")
        
        # Radar Chart
        categories = ['Glucose', 'Blood Pressure', 'BMI', 'Insulin (Scaled)']
        vis_insulin = min(insulin, 200) 
        healthy_values = [100, 70, 22, 80]
        patient_values = [glucose, bp, bmi, vis_insulin]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=patient_values, 
            theta=categories, 
            fill='toself', 
            name='Patient Stats', 
            line_color='#667eea',
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=healthy_values, 
            theta=categories, 
            fill='toself', 
            name='Healthy Average', 
            line_color='#48bb78',
            fillcolor='rgba(72, 187, 120, 0.2)'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, max(max(patient_values), 200)],
                    gridcolor='#e2e8f0'
                ),
                bgcolor='rgba(255, 255, 255, 0.5)'
            ),
            showlegend=True,
            height=450,
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            font={'family': 'Poppins, sans-serif', 'color': '#2d3748'}
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        st.header("📋 Comprehensive Care Plan")
        
        col_diet, col_lifestyle = st.columns(2)
        
        with col_diet:
            st.subheader("🥗 Dietary Guidelines")
            st.markdown("""
            **Focus on Low Glycemic Index (GI) Foods:**
            * **Vegetables:** Leafy greens (spinach, kale), broccoli, cauliflower
            * **Proteins:** Fish, skinless chicken, tofu, lentils, beans
            * **Healthy Fats:** Avocados, nuts (walnuts, almonds), olive oil
            * **Whole Grains:** Quinoa, oats, brown rice (in moderation)
            """)
            
            st.error("""
            **🚫 Foods to Limit/Avoid:**
            * Refined sugars (soda, candy, pastries)
            * White bread, white pasta, and white rice
            * Processed meats (sausages, bacon)
            * Fried foods and trans fats
            """)

        with col_lifestyle:
            st.subheader("🏃‍♂️ Lifestyle Habits")
            st.markdown("""
            **Daily Routine:**
            * **Exercise:** Aim for at least 30 minutes of moderate activity (brisk walking, swimming) 5 days a week
            * **Sleep:** Prioritize 7-9 hours of quality sleep to regulate hormones
            * **Hydration:** Drink 8-10 glasses of water daily
            """)
            
            if risk_category == "High":
                st.warning("""
                **⚠️ Urgent Lifestyle Changes:**
                * Monitor blood sugar daily
                * Stop smoking immediately if applicable
                * Consult a nutritionist for a personalized meal plan
                """)
            else:
                st.info("""
                **✅ Preventive Measures:**
                * Maintain current weight
                * Annual screening for blood sugar and cholesterol
                """)

        # --- Disclaimer ---
        st.markdown("---")
        st.warning("""
        **⚠️ MEDICAL DISCLAIMER**
        
        The dietary and lifestyle recommendations provided above are general guidelines generated by AI. 
        They do not replace professional medical advice, diagnosis, or treatment. 
        **Always seek the advice of your physician or qualified health provider** regarding your specific medical condition.
        """)
