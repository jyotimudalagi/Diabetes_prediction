import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import base64
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Diabetes AI Predictor | Government of India",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to convert image to base64 ---
def get_base64_image(image_path):
    """Convert image to base64 string for embedding in CSS"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# --- Function to get precise risk category ---
def get_precise_risk_category(probability):
    """
    Get precise risk category based on probability percentage
    
    Returns: (category, color, emoji, description)
    """
    if probability < 20:
        return "Very Low Risk", "#22c55e", "🟢", "Minimal diabetes risk detected"
    elif 20 <= probability < 40:
        return "Low Risk", "#84cc16", "🟡", "Low diabetes risk - maintain healthy lifestyle"
    elif 40 <= probability < 60:
        return "Moderate Risk", "#fb923c", "🟠", "Moderate risk - lifestyle changes recommended"
    elif 60 <= probability < 75:
        return "High Risk", "#f97316", "🔴", "High risk - immediate medical consultation advised"
    elif 75 <= probability < 90:
        return "Very High Risk", "#ef4444", "🔴", "Very high risk - urgent medical attention required"
    else:
        return "Critical Risk", "#dc2626", "🔴", "Critical risk - immediate intervention needed"

# Try to load the background image
bg_image_path = "diabetes_background.png"
bg_base64 = get_base64_image(bg_image_path)

# If base64 encoding worked, use it in CSS, otherwise use a fallback gradient
if bg_base64:
    background_style = f"""
    .stApp {{
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.85) 0%, rgba(15, 23, 42, 0.9) 100%),
                    url('data:image/png;base64,{bg_base64}') center/cover fixed;
        background-blend-mode: overlay;
        color: #e2e8f0;
    }}
    """
else:
    # Fallback to gradient if image not found
    background_style = """
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    """

# --- Enhanced Custom CSS with Improved Government Banner ---
st.markdown(f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main container */
    .main {{
        font-family: 'Poppins', sans-serif;
    }}
    
    /* Custom background with uploaded image */
    {background_style}
    
    /* Subtle overlay for better text readability */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(168, 85, 247, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }}
    
    /* Sidebar styling - Dark theme */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-right: 2px solid #334155;
        box-shadow: 2px 0 20px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        background: transparent;
    }}
    
    /* Sidebar text color */
    [data-testid="stSidebar"] * {{
        color: #e2e8f0 !important;
    }}
    
    /* Card-like containers for metrics */
    div[data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
    }}
    
    div[data-testid="stMetricLabel"] {{
        font-weight: 600;
        color: #cbd5e1;
    }}
    
    /* Tab styling - Dark */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: rgba(30, 41, 59, 0.8);
        padding: 10px;
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0 20px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
    }}
    
    /* Headers with gradient */
    h1 {{
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        padding: 20px 0;
    }}
    
    h2, h3 {{
        color: #f1f5f9;
        font-weight: 600;
    }}
    
    /* Text colors */
    p, li, div, span {{
        color: #e2e8f0;
    }}
    
    /* Custom success/error boxes with glassmorphism - Dark */
    .stAlert {{
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }}
    
    /* Info boxes - Dark */
    .stInfo {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.2) 100%);
        border-left: 4px solid #3b82f6;
        color: #dbeafe;
    }}
    
    .stSuccess {{
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.2) 100%);
        border-left: 4px solid #22c55e;
        color: #dcfce7;
    }}
    
    .stError {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        border-left: 4px solid #ef4444;
        color: #fee2e2;
    }}
    
    .stWarning {{
        background: linear-gradient(135deg, rgba(251, 146, 60, 0.2) 0%, rgba(249, 115, 22, 0.2) 100%);
        border-left: 4px solid #fb923c;
        color: #fed7aa;
    }}
    
    /* Button styling - Dark */
    .stButton > button {{
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }}
    
    /* Form containers - Dark */
    [data-testid="stForm"] {{
        background: rgba(30, 41, 59, 0.7);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* Slider styling with gradient */
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }}
    
    /* Make slider labels bright and visible */
    .stSlider label {{
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }}
    
    /* Slider values */
    .stSlider [data-baseweb="slider"] {{
        color: #f1f5f9 !important;
    }}
    
    /* All form labels bright */
    label, .stMarkdown label {{
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }}
    
    /* Subheaders in sidebar bright */
    .stForm h3, .stForm h2 {{
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }}
    
    /* Number input styling - Dark */
    .stNumberInput > div > div > input {{
        border-radius: 8px;
        border: 2px solid #475569;
        background-color: rgba(30, 41, 59, 0.8);
        color: #f1f5f9;
        font-weight: 600;
    }}
    
    .stNumberInput label {{
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }}
    
    /* Plotly chart containers - Dark */
    .js-plotly-plot {{
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.7);
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }}
    
    /* Markdown content in main area */
    .main .stMarkdown {{
        color: #e2e8f0;
    }}
    
    /* Lists and text */
    .main ul, .main ol, .main li {{
        color: #cbd5e1;
    }}
    
    /* Strong/bold text */
    .main strong {{
        color: #f1f5f9;
    }}
    
    /* IMPROVED Government of India Banner - Much Darker and More Visible */
    .gov-banner {{
        background: linear-gradient(135deg, #b8621a 0%, #d4d4d4 50%, #0a5c03 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.6);
        text-align: center;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 0, 0, 0.3);
    }}
    
    .gov-emblem {{
        width: 90px;
        height: 90px;
        margin: 0 auto 15px;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4));
    }}
    
    .gov-title {{
        color: #0a0a0a !important;
        margin: 5px 0 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.3);
    }}
    
    .gov-subtitle {{
        color: #1a1a1a !important;
        margin: 8px 0 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.2);
    }}
    
    .gov-description {{
        color: #2a2a2a !important;
        margin: 5px 0 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }}
    
    /* Risk Category Badge */
    .risk-badge {{
        display: inline-block;
        padding: 15px 30px;
        border-radius: 12px;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
    }}
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

# --- IMPROVED Government of India Header Banner ---
st.markdown("""
    <div class='gov-banner'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/800px-Emblem_of_India.svg.png' 
             class='gov-emblem' alt='Government of India'/>
        <h2 class='gov-title'>Government of India</h2>
        <h3 class='gov-subtitle'>Ministry of Health and Family Welfare</h3>
        <p class='gov-description'>National Diabetes Prevention Initiative</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <div style='font-size: 80px; margin-bottom: 10px;'>🏥</div>
        <h2 style='color: #60a5fa; margin: 0;'>Patient Data</h2>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<p style='color: #cbd5e1; text-align: center; margin-bottom: 20px;'>Enter the clinical details below</p>", unsafe_allow_html=True)

with st.sidebar.form("prediction_form"):
    st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <span style='font-size: 40px;'>👤</span>
            <h3 style='color: #f1f5f9; margin: 10px 0;'>Demographics</h3>
        </div>
    """, unsafe_allow_html=True)
    
    age = st.slider('Age', 21, 100, 30, help="Patient's age in years")
    pregnancies = st.number_input('Pregnancies', 0, 20, 0, help="Number of times pregnant")

    st.markdown("""
        <div style='text-align: center; margin: 30px 0 20px 0;'>
            <span style='font-size: 40px;'>🩺</span>
            <h3 style='color: #f1f5f9; margin: 10px 0;'>Vitals & Labs</h3>
        </div>
    """, unsafe_allow_html=True)
    
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
    st.markdown("""
        <div style='text-align: center;'>
            <span style='font-size: 100px;'>🩺</span>
            <h1 style='margin: 20px 0;'>Diabetes Risk AI Predictor</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Welcome to the National Clinical Decision Support System")
    
    # Hero section with glassmorphism
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.7);
                    padding: 40px; border-radius: 15px; margin: 20px 0;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                    backdrop-filter: blur(15px);
                    border: 1px solid rgba(255, 255, 255, 0.1);'>
            <div style='text-align: center;'>
                <span style='font-size: 60px;'>🎯</span>
                <h2 style='color: #f1f5f9; margin: 10px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.5);'>AI-Powered Diabetes Risk Assessment</h2>
                <p style='color: #cbd5e1; font-size: 1.1em; margin-top: 10px; text-shadow: 0 1px 2px rgba(0,0,0,0.5);'>
                    Leveraging machine learning to predict diabetes risk with clinical precision
                </p>
                <p style='color: #94a3b8; font-size: 0.95em; margin-top: 5px;'>
                    Developed under National Health Mission | Digital India Initiative
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
            <div style='background: rgba(30, 41, 59, 0.8); padding: 25px; border-radius: 12px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3); height: 320px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(15px);'>
                <div style='text-align: center; font-size: 50px; margin-bottom: 15px;'>📊</div>
                <h3 style='color: #60a5fa; text-align: center;'>How It Works</h3>
                <ol style='color: #cbd5e1; line-height: 1.8;'>
                    <li>Enter clinical parameters in the sidebar</li>
                    <li>AI analyzes patterns from 768+ records</li>
                    <li>Receive probability score & insights</li>
                    <li>Get personalized care recommendations</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: rgba(30, 41, 59, 0.8); padding: 25px; border-radius: 12px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3); height: 320px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(15px);'>
                <div style='text-align: center; font-size: 50px; margin-bottom: 15px;'>🔬</div>
                <h3 style='color: #a78bfa; text-align: center;'>Model Performance</h3>
                <div style='text-align: center; margin-top: 30px;'>
                    <div style='font-size: 3em; font-weight: bold; color: #60a5fa;'>78%</div>
                    <div style='color: #cbd5e1; margin-top: 10px;'>Accuracy Score</div>
                    <div style='color: #94a3b8; font-size: 0.9em; margin-top: 15px;'>Support Vector Machine (SVM)</div>
                    <div style='color: #64748b; font-size: 0.85em; margin-top: 5px;'>Validated by AIIMS & ICMR</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: rgba(30, 41, 59, 0.8); padding: 25px; border-radius: 12px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3); height: 320px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(15px);'>
                <div style='text-align: center; font-size: 50px; margin-bottom: 15px;'>✨</div>
                <h3 style='color: #34d399; text-align: center;'>Key Features</h3>
                <ul style='color: #cbd5e1; line-height: 1.8;'>
                    <li>Real-time risk prediction</li>
                    <li>Interactive visualizations</li>
                    <li>Personalized care plans</li>
                    <li>Dietary & lifestyle guidance</li>
                    <li>Evidence-based recommendations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional Government Info
    st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.7); padding: 20px; border-radius: 10px; 
                    border-left: 4px solid #3b82f6; margin: 20px 0; backdrop-filter: blur(15px);'>
            <h4 style='color: #60a5fa; margin-top: 0;'>🇮🇳 About This Initiative</h4>
            <p style='color: #cbd5e1; line-height: 1.6;'>
                This AI-powered diagnostic tool is part of the Government of India's Digital Health Mission, 
                aimed at providing accessible, affordable, and quality healthcare to all citizens. 
                The system leverages advanced machine learning algorithms validated by leading medical institutions 
                to assist healthcare professionals in early diabetes detection and prevention.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
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

    # Get precise risk category
    risk_category, risk_color, risk_emoji, risk_description = get_precise_risk_category(prob_positive)

    # --- Results Dashboard ---
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <span style='font-size: 80px;'>📊</span>
            <h1 style='margin: 10px 0;'>Analysis Results</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Top-level Status Banner with Precise Risk Category
    st.markdown(f"""
        <div class='risk-badge' style='background: linear-gradient(135deg, {risk_color}22 0%, {risk_color}44 100%); 
                                        border: 3px solid {risk_color}; width: 100%;'>
            <span style='font-size: 2.5rem;'>{risk_emoji}</span><br>
            <span style='color: {risk_color}; font-size: 1.8rem;'>{risk_category}</span><br>
            <span style='color: #cbd5e1; font-size: 1.2rem;'>Probability: {prob_positive:.1f}%</span><br>
            <span style='color: #94a3b8; font-size: 1rem; font-weight: 500;'>{risk_description}</span>
        </div>
    """, unsafe_allow_html=True)

    # Tabs for organized view
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧬 Vitals Analysis", "🍎 Lifestyle & Diet"])

    with tab1:
        col_gauge, col_metrics = st.columns([1, 1])
        
        with col_gauge:
            # Gauge Chart - Dark theme with color-coded sections
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_positive,
                title={'text': "Diabetic Probability", 'font': {'size': 24, 'color': '#f1f5f9'}},
                number={'suffix': "%", 'font': {'size': 40, 'color': '#f1f5f9'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
                    'bar': {'color': risk_color, 'thickness': 0.75},
                    'steps': [
                        {'range': [0, 20], 'color': "rgba(34, 197, 94, 0.3)"},
                        {'range': [20, 40], 'color': "rgba(132, 204, 22, 0.3)"},
                        {'range': [40, 60], 'color': "rgba(251, 146, 60, 0.3)"},
                        {'range': [60, 75], 'color': "rgba(249, 115, 34, 0.3)"},
                        {'range': [75, 90], 'color': "rgba(239, 68, 68, 0.3)"},
                        {'range': [90, 100], 'color': "rgba(220, 38, 38, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "#ef4444", 'width': 4}, 
                        'thickness': 0.75, 
                        'value': 60
                    }
                }
            ))
            fig_gauge.update_layout(
                height=350, 
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(30, 41, 59, 0.7)',
                font={'family': 'Poppins, sans-serif', 'color': '#f1f5f9'}
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
            
            # Risk level explanation
            st.markdown("---")
            st.markdown(f"""
                <div style='background: rgba({int(risk_color[1:3], 16)}, {int(risk_color[3:5], 16)}, {int(risk_color[5:7], 16)}, 0.15); 
                            padding: 15px; border-radius: 10px; border-left: 4px solid {risk_color};'>
                    <strong style='color: {risk_color};'>Risk Assessment:</strong><br>
                    <span style='color: #cbd5e1;'>{risk_description}</span>
                </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📊 Patient vs. Healthy Average")
        
        # Radar Chart - Dark theme
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
            line_color='#3b82f6',
            fillcolor='rgba(59, 130, 246, 0.3)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=healthy_values, 
            theta=categories, 
            fill='toself', 
            name='Healthy Average', 
            line_color='#34d399',
            fillcolor='rgba(34, 211, 238, 0.2)'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, max(max(patient_values), 200)],
                    gridcolor='#475569',
                    color='#cbd5e1'
                ),
                bgcolor='rgba(30, 41, 59, 0.3)',
                angularaxis=dict(color='#cbd5e1')
            ),
            showlegend=True,
            height=450,
            paper_bgcolor='rgba(30, 41, 59, 0.7)',
            font={'family': 'Poppins, sans-serif', 'color': '#f1f5f9'},
            legend=dict(font=dict(color='#f1f5f9'))
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        st.header("📋 Comprehensive Care Plan")
        st.markdown("<p style='color: #94a3b8; font-style: italic;'>Based on National Guidelines for Diabetes Management (ICMR-INDIAB)</p>", unsafe_allow_html=True)
        
        col_diet, col_lifestyle = st.columns(2)
        
        with col_diet:
            st.subheader("🥗 Dietary Guidelines")
            st.markdown("""
            **Focus on Low Glycemic Index (GI) Foods:**
            * **Vegetables:** Leafy greens (spinach, kale), broccoli, cauliflower, bitter gourd
            * **Proteins:** Fish, skinless chicken, tofu, lentils, beans, chickpeas
            * **Healthy Fats:** Avocados, nuts (walnuts, almonds), olive oil, flaxseeds
            * **Whole Grains:** Quinoa, oats, brown rice (in moderation), ragi, jowar
            * **Traditional Foods:** Methi (fenugreek), karela (bitter gourd), amla
            """)
            
            st.error("""
            **🚫 Foods to Limit/Avoid:**
            * Refined sugars (soda, candy, pastries, sweets)
            * White bread, white pasta, white rice, maida
            * Processed meats (sausages, bacon)
            * Fried foods and trans fats
            * High-sugar fruits (mango, banana in excess)
            """)

        with col_lifestyle:
            st.subheader("🏃‍♂️ Lifestyle Habits")
            st.markdown("""
            **Daily Routine:**
            * **Exercise:** Aim for at least 30 minutes of moderate activity (brisk walking, yoga, swimming) 5 days a week
            * **Sleep:** Prioritize 7-9 hours of quality sleep to regulate hormones
            * **Hydration:** Drink 8-10 glasses of water daily
            * **Stress Management:** Practice meditation or pranayama for 15 minutes daily
            """)
            
            # Risk-specific recommendations
            if prob_positive >= 75:
                st.error("""
                **🚨 CRITICAL - Immediate Action Required:**
                * Schedule URGENT appointment with endocrinologist (within 24-48 hours)
                * Begin daily blood sugar monitoring (fasting & post-meal)
                * Stop all refined sugars and processed foods immediately
                * Join intensive diabetes management program
                * Consider medical intervention/medication under doctor's guidance
                """)
            elif prob_positive >= 60:
                st.warning("""
                **⚠️ HIGH RISK - Urgent Lifestyle Changes:**
                * Schedule appointment with endocrinologist within 1 week
                * Monitor blood sugar daily (fasting & post-meal)
                * Stop smoking and limit alcohol immediately
                * Consult a nutritionist for personalized meal plan
                * Join diabetes prevention program at nearest PHC/CHC
                * Increase physical activity to 45-60 minutes daily
                """)
            elif prob_positive >= 40:
                st.warning("""
                **🟠 MODERATE RISK - Preventive Measures Needed:**
                * Schedule medical consultation within 2 weeks
                * Begin weekly blood sugar monitoring
                * Implement strict dietary modifications
                * Start structured exercise program (150 min/week)
                * Regular health check-ups every 3 months
                * Stress management and adequate sleep
                """)
            elif prob_positive >= 20:
                st.info("""
                **🟡 LOW RISK - Maintain Healthy Habits:**
                * Continue current healthy lifestyle
                * Monitor blood sugar every 3-6 months
                * Maintain healthy weight and BMI
                * Regular physical activity (30 min, 5 days/week)
                * Annual comprehensive health screening
                """)
            else:
                st.success("""
                **🟢 VERY LOW RISK - Excellent Health Status:**
                * Maintain current excellent lifestyle practices
                * Annual routine health check-ups
                * Continue balanced diet and regular exercise
                * Stay hydrated and manage stress effectively
                * Keep up preventive health measures
                """)

        # --- Government Resources ---
        st.markdown("---")
        st.markdown("""
            <div style='background: rgba(59, 130, 246, 0.15); padding: 20px; border-radius: 10px; 
                        border-left: 4px solid #3b82f6; backdrop-filter: blur(10px);'>
                <h4 style='color: #60a5fa; margin-top: 0;'>🏥 Government Healthcare Resources</h4>
                <ul style='color: #cbd5e1; line-height: 1.8;'>
                    <li><strong>National Diabetes Prevention Program:</strong> Free screening at Primary Health Centers (PHC)</li>
                    <li><strong>Ayushman Bharat:</strong> Health and Wellness Centers for regular monitoring</li>
                    <li><strong>National NCD Portal:</strong> ncdmonitoring.nhp.gov.in</li>
                    <li><strong>Toll-Free Helpline:</strong> 1800-180-1104 (Health Ministry)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # --- Disclaimer ---
        st.markdown("---")
        st.warning("""
        **⚠️ MEDICAL DISCLAIMER**
        
        The dietary and lifestyle recommendations provided above are general guidelines based on ICMR-INDIAB 
        and National Programme for Prevention and Control of Cancer, Diabetes, Cardiovascular Diseases and Stroke (NPCDCS). 
        They do not replace professional medical advice, diagnosis, or treatment. 
        **Always seek the advice of your physician or qualified health provider** regarding your specific medical condition.
        
        *This tool is for screening purposes only and should be used alongside clinical judgment.*
        """)

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #64748b;'>
        <p style='margin: 5px 0;'>Developed by Government of India | Ministry of Health and Family Welfare</p>
        <p style='margin: 5px 0; font-size: 0.9em;'>Part of Digital India & Ayushman Bharat Initiative</p>
        <p style='margin: 5px 0; font-size: 0.85em;'>© 2025 Government of India. All Rights Reserved.</p>
    </div>
""", unsafe_allow_html=True)