import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="AI Bean Classification System",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        color: white;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.3rem;
        font-weight: 400;
        line-height: 1.6;
        opacity: 0.95;
    }
    
    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2f3542;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Form styling */
    .form-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .form-section-title {
        color: #2f3542;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #ff6b6b;
        display: inline-block;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border: 2px solid #e1e5e9;
        border-radius: 12px;
        padding: 12px 16px;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #ff6b6b;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.1);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.3rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(255, 107, 107, 0.6);
    }
    
    /* Result styling */
    .prediction-result {
        background: linear-gradient(135deg, #96ceb4, #feca57);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(150, 206, 180, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-result::before {
        content: '🫘';
        position: absolute;
        top: 20px;
        right: 30px;
        font-size: 8rem;
        opacity: 0.1;
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Metrics styling */
    .metrics-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info boxes */
    .info-highlight {
        background: linear-gradient(135deg, rgba(69, 183, 209, 0.1), rgba(78, 205, 196, 0.1));
        border-left: 5px solid #45b7d1;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .warning-highlight {
        background: linear-gradient(135deg, rgba(254, 202, 87, 0.1), rgba(255, 107, 107, 0.1));
        border-left: 5px solid #feca57;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load('svc_best_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file 'svc_best_model.pkl' not found! Please ensure it's in the app directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

model_SVC = load_model()

# -------------------- Bean Configuration --------------------
class_mapping = {
    "DERMASON": 0,
    "SIRA": 1,
    "SEKER": 2,
    "HOROZ": 3,
    "CALI": 4,
    "BARBUNYA": 5,
    "BOMBAY": 6
}

reverse_class_mapping = {v: k for k, v in class_mapping.items()}

bean_emojis = {
    "DERMASON": "🟤",
    "SIRA": "🟡", 
    "SEKER": "⚪",
    "HOROZ": "🔴",
    "CALI": "🟢",
    "BARBUNYA": "🟣",
    "BOMBAY": "⚫"
}

features = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

# -------------------- Header --------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🫘 AI Bean Classification System</div>
    <div class="main-subtitle">
        Advanced machine learning for precise bean type identification<br>
        <strong>7 Bean Types</strong> • <strong>Instant Results</strong> • <strong>High Accuracy</strong>
    </div>
    <div style="margin-top: 1.5rem; font-size: 1rem; opacity: 0.8;">
        <strong>Powered by Support Vector Machine Algorithm</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Feature Cards --------------------
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <span class="feature-icon">🤖</span>
        <div class="feature-title">AI-Powered</div>
        <div class="feature-desc">Advanced SVM algorithm for accurate classification</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <div class="feature-title">Lightning Fast</div>
        <div class="feature-desc">Get instant bean type predictions in seconds</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <div class="feature-title">Data-Driven</div>
        <div class="feature-desc">Based on comprehensive geometric analysis</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🔒</span>
        <div class="feature-title">Reliable</div>
        <div class="feature-desc">Trusted classification you can count on</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### 🎯 Quick Guide")
    st.markdown("""
    <div class="info-highlight">
        <strong>How to classify beans:</strong><br>
        1. Enter bean feature measurements<br>
        2. Input geometric and shape parameters<br>
        3. Click classify for instant AI prediction<br>
        4. Get accurate bean type identification
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🫘 Bean Types")
    st.markdown("""
    <div class="warning-highlight">
        <strong>Classifiable bean varieties:</strong><br>
        🟤 <strong>DERMASON</strong><br>
        🟡 <strong>SIRA</strong><br>
        ⚪ <strong>SEKER</strong><br>
        🔴 <strong>HOROZ</strong><br>
        🟢 <strong>CALI</strong><br>
        🟣 <strong>BARBUNYA</strong><br>
        ⚫ <strong>BOMBAY</strong>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Main Form --------------------
st.markdown("""
<div class="form-container">
    <h2 class="form-section-title">🔧 Bean Feature Measurements</h2>
</div>
""", unsafe_allow_html=True)

# Create improved layout
col1, col2 = st.columns([1, 1])

values = []

with col1:
    st.markdown("#### 📐 Geometric Features")
    Area = st.number_input("📏 Area", min_value=0.0, value=28395.0, format="%.2f", help="Surface area of the bean")
    Perimeter = st.number_input("⭕ Perimeter", min_value=0.0, value=610.0, format="%.2f", help="Perimeter length")
    MajorAxisLength = st.number_input("↔️ Major Axis Length", min_value=0.0, value=208.0, format="%.2f", help="Length of major axis")
    MinorAxisLength = st.number_input("↕️ Minor Axis Length", min_value=0.0, value=173.0, format="%.2f", help="Length of minor axis")
    AspectRation = st.number_input("📊 Aspect Ratio", min_value=0.0, value=1.2, format="%.4f", help="Ratio of major to minor axis")
    Eccentricity = st.number_input("🌀 Eccentricity", min_value=0.0, max_value=1.0, value=0.5, format="%.4f", help="Measure of deviation from circle")
    ConvexArea = st.number_input("📦 Convex Area", min_value=0.0, value=28715.0, format="%.2f", help="Area of convex hull")
    EquivDiameter = st.number_input("⚫ Equiv Diameter", min_value=0.0, value=190.0, format="%.2f", help="Equivalent circle diameter")
    
    values.extend([Area, Perimeter, MajorAxisLength, MinorAxisLength, 
                   AspectRation, Eccentricity, ConvexArea, EquivDiameter])

with col2:
    st.markdown("#### 🔍 Shape Features")
    Extent = st.number_input("📐 Extent", min_value=0.0, max_value=1.0, value=0.7, format="%.4f", help="Ratio of region area to bounding box")
    Solidity = st.number_input("💎 Solidity", min_value=0.0, max_value=1.0, value=0.98, format="%.4f", help="Ratio of region area to convex area")
    roundness = st.number_input("⭕ Roundness", min_value=0.0, value=0.8, format="%.4f", help="Measure of roundness")
    Compactness = st.number_input("📦 Compactness", min_value=0.0, value=0.8, format="%.4f", help="Measure of compactness")
    ShapeFactor1 = st.number_input("🔢 Shape Factor 1", min_value=0.0, value=0.006, format="%.6f", help="First shape factor")
    ShapeFactor2 = st.number_input("🔢 Shape Factor 2", min_value=0.0, value=0.001, format="%.6f", help="Second shape factor")
    ShapeFactor3 = st.number_input("🔢 Shape Factor 3", min_value=0.0, value=0.8, format="%.4f", help="Third shape factor")
    ShapeFactor4 = st.number_input("🔢 Shape Factor 4", min_value=0.0, value=0.98, format="%.4f", help="Fourth shape factor")
    
    values.extend([Extent, Solidity, roundness, Compactness, 
                   ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4])

# -------------------- Real-time Metrics --------------------
st.markdown("---")
st.markdown("### 📊 Feature Analysis")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    shape_quality = "Round" if roundness > 0.7 else "Oval" if roundness > 0.5 else "Elongated"
    shape_color = "🟢" if roundness > 0.7 else "🟡" if roundness > 0.5 else "🔴"
    st.metric("🔵 Shape", f"{shape_color} {shape_quality}")

with metric_col2:
    size_level = "Large" if Area > 50000 else "Medium" if Area > 25000 else "Small"
    size_color = "🟢" if Area > 50000 else "🟡" if Area > 25000 else "🔴"
    st.metric("📏 Size", f"{size_color} {size_level}")

with metric_col3:
    density = "High" if Solidity > 0.95 else "Medium" if Solidity > 0.90 else "Low"
    density_color = "🟢" if Solidity > 0.95 else "🟡" if Solidity > 0.90 else "🔴"
    st.metric("💎 Solidity", f"{density_color} {density}")

with metric_col4:
    compactness_level = "High" if Compactness > 0.8 else "Medium" if Compactness > 0.6 else "Low"
    compact_color = "🟢" if Compactness > 0.8 else "🟡" if Compactness > 0.6 else "🔴"
    st.metric("📦 Compact", f"{compact_color} {compactness_level}")

# -------------------- Prediction Section --------------------
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    if st.button("🔮 **CLASSIFY BEAN TYPE**", key="predict_btn", help="Click to get instant AI-powered classification"):
        if model_SVC is None:
            st.error("❌ Model not available. Please check the model file.")
        else:
            try:
                # Create DataFrame
                df = pd.DataFrame([values], columns=features)
                
                # Make prediction
                with st.spinner("🤖 AI is analyzing bean features..."):
                    prediction_encoded = model_SVC.predict(df)[0]
                    bean_name = reverse_class_mapping.get(prediction_encoded, f"Unknown ({prediction_encoded})")
                    bean_emoji = bean_emojis.get(bean_name, "🫘")
                    
                    # Display result with animation
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            🎉 Predicted Bean Type
                        </h2>
                        <div class="prediction-value">{bean_emoji} {bean_name}</div>
                        <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                            Based on geometric and shape feature analysis
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probability analysis
                    st.markdown("### 💡 Classification Analysis")
                    
                    # Try to get probabilities
                    try:
                        probabilities = model_SVC.predict_proba(df)[0]
                        max_prob = np.max(probabilities)
                        
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            st.markdown(f"""
                            <div class="metrics-container">
                                <h4>🎯 Confidence Score</h4>
                                <div class="metric-card">
                                    <div class="metric-value">{max_prob:.1%}</div>
                                    <div class="metric-label">Prediction Confidence</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with analysis_col2:
                            # Get top 3 predictions
                            top_3_indices = np.argsort(probabilities)[-3:][::-1]
                            top_3_beans = [reverse_class_mapping.get(i, f"Class {i}") for i in top_3_indices]
                            top_3_probs = [probabilities[i] for i in top_3_indices]
                            
                            st.markdown(f"""
                            <div class="metrics-container">
                                <h4>🏆 Top Predictions</h4>
                                <div class="metric-card">
                                    <div class="metric-label">
                                        1. {bean_emojis.get(top_3_beans[0], '🫘')} {top_3_beans[0]}: {top_3_probs[0]:.1%}<br>
                                        2. {bean_emojis.get(top_3_beans[1], '🫘')} {top_3_beans[1]}: {top_3_probs[1]:.1%}<br>
                                        3. {bean_emojis.get(top_3_beans[2], '🫘')} {top_3_beans[2]}: {top_3_probs[2]:.1%}
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Create probability chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        bean_types = [reverse_class_mapping.get(i, f"Class {i}") for i in range(len(probabilities))]
                        bean_labels = [f"{bean_emojis.get(bt, '🫘')} {bt}" for bt in bean_types]
                        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9a9e', '#a29bfe']
                        
                        bars = ax.barh(bean_labels, probabilities * 100, color=colors[:len(bean_types)], alpha=0.8)
                        
                        # Add value labels on bars
                        for bar, prob in zip(bars, probabilities):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2., 
                                   f'{prob*100:.1f}%', 
                                   ha='left', va='center', fontsize=11, fontweight='bold')
                        
                        ax.set_xlabel('Probability (%)', fontsize=12)
                        ax.set_title('🎯 Bean Type Probability Distribution', fontsize=16, fontweight='bold', pad=20)
                        ax.set_xlim(0, max(probabilities) * 110)
                        
                        # Style the plot
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.grid(axis='x', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                    except Exception as e:
                        st.info("💡 Probability distribution not available for this model")
                    
                    # Success animation
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Classification failed: {str(e)}")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 3rem; backdrop-filter: blur(10px);">
    <h3 style="color: white; margin-bottom: 1rem;">🚀 Powered by Advanced Machine Learning</h3>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; line-height: 1.6; margin: 0;">
        Our Support Vector Machine model analyzes 16 distinct features including geometric properties, 
        shape factors, and derived metrics to accurately classify 7 different bean varieties with high precision.
    </p>
    <br>
    <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">Advanced Bean Classification</p>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Machine Learning in Agricultural Technology
        </p>
    </div>
    <p style="color: rgba(255,255,255,0.7); font-style: italic; margin: 0;">
        <strong>Note:</strong> Ensure accurate feature measurements for best classification results.
        All measurements should be in standardized units.
    </p>
</div>
""", unsafe_allow_html=True)



