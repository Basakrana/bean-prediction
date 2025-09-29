import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Bean Classification System",
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
    
    /* Sidebar styling */
    .sidebar-bean {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        text-align: center;
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

model = load_model()

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
    <div class="main-title">🫘 Bean Classification System</div>
    <div class="main-subtitle">
        Advanced AI-powered bean type recognition using machine learning<br>
        <strong>7 Bean Types</strong> • <strong>Instant Results</strong> • <strong>High Accuracy</strong>
    </div>
    <div style="margin-top: 1.5rem; font-size: 1rem; opacity: 0.8;">
        <strong>Powered by Support Vector Machine</strong>
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
        <div class="feature-desc">Get instant bean type predictions</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <div class="feature-title">16 Features</div>
        <div class="feature-desc">Comprehensive geometric and shape analysis</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🎯</span>
        <div class="feature-title">7 Bean Types</div>
        <div class="feature-desc">Classifies multiple bean varieties</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### 🫘 Bean Types")
    st.markdown("""
    <div class="info-highlight">
        <strong>This system classifies:</strong>
    </div>
    """, unsafe_allow_html=True)
    
    for bean_name, code in class_mapping.items():
        emoji = bean_emojis.get(bean_name, "🫘")
        st.markdown(f"""
        <div class="sidebar-bean">
            {emoji} {bean_name}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📏 Features Used")
    st.markdown("""
    <div class="info-highlight">
        <strong>Feature Categories:</strong><br>
        • <strong>Geometric:</strong> Area, Perimeter, Axes<br>
        • <strong>Shape:</strong> Roundness, Compactness<br>
        • <strong>Derived:</strong> Shape Factors 1-4
    </div>
    """, unsafe_allow_html=True)

# -------------------- Input Method Selection --------------------
st.markdown("""
<div class="form-container">
    <h2 class="form-section-title">🔧 Choose Input Method</h2>
</div>
""", unsafe_allow_html=True)

input_method = st.radio("", ["🔢 Manual Input", "📁 CSV Upload"], horizontal=True, label_visibility="collapsed")

if "Manual" in input_method:
    # -------------------- Manual Input --------------------
    st.markdown("""
    <div class="form-container">
        <h2 class="form-section-title">📊 Enter Bean Features</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    values = []
    geometric_features = features[:8]
    shape_features = features[8:]
    
    with col1:
        st.markdown("#### 📐 Geometric Features")
        for feature in geometric_features:
            value = st.number_input(f"{feature}", value=0.0, format="%.6f", key=feature)
            values.append(value)
    
    with col2:
        st.markdown("#### 🔍 Shape Features")
        for feature in shape_features:
            value = st.number_input(f"{feature}", value=0.0, format="%.6f", key=feature)
            values.append(value)
    
    # -------------------- Prediction Button --------------------
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        if st.button("🎯 **CLASSIFY BEAN**", key="predict_btn"):
            if model is None:
                st.error("❌ Model not available. Please check the model file.")
            else:
                try:
                    input_data = pd.DataFrame([values], columns=features)
                    
                    with st.spinner("🤖 AI is analyzing bean features..."):
                        prediction_encoded = model.predict(input_data)[0]
                        bean_name = reverse_class_mapping.get(prediction_encoded, f"Unknown ({prediction_encoded})")
                        bean_emoji = bean_emojis.get(bean_name, "🫘")
                        
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h2 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                🎉 Predicted Bean Type
                            </h2>
                            <div class="prediction-value">{bean_emoji} {bean_name}</div>
                            <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                                Classification complete
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show probabilities if available
                        try:
                            probabilities = model.predict_proba(input_data)[0]
                            max_prob = np.max(probabilities)
                            
                            st.markdown(f"""
                            <div style="text-align: center; margin: 1rem 0;">
                                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; color: white; font-weight: 600;">
                                    Confidence: {max_prob:.1%}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Create probability chart
                            st.markdown("### 📊 Probability Distribution")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            bean_types = [reverse_class_mapping.get(i, f"Class {i}") for i in range(len(probabilities))]
                            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9a9e', '#a29bfe']
                            
                            bars = ax.barh(bean_types, probabilities * 100, color=colors[:len(bean_types)], alpha=0.8)
                            
                            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                                width = bar.get_width()
                                emoji = bean_emojis.get(bean_types[i], "🫘")
                                ax.text(width + 1, bar.get_y() + bar.get_height()/2., 
                                       f'{emoji} {prob*100:.1f}%', 
                                       ha='left', va='center', fontsize=11, fontweight='bold')
                            
                            ax.set_xlabel('Probability (%)', fontsize=12)
                            ax.set_title('🎯 Bean Type Probability Distribution', fontsize=16, fontweight='bold', pad=20)
                            ax.set_xlim(0, max(probabilities) * 110)
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.grid(axis='x', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                        except Exception as e:
                            st.info("💡 Probability distribution not available for this model")
                        
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

else:
    # -------------------- CSV Upload --------------------
    st.markdown("""
    <div class="form-container">
        <h2 class="form-section-title">📁 Batch Classification</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"📋 **Required columns:** {', '.join(features[:4])}... (total: {len(features)} columns)")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("**📊 Data Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            missing_cols = [col for col in features if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
                with predict_col2:
                    if st.button("🎯 **CLASSIFY ALL BEANS**", key="batch_predict_btn"):
                        if model is None:
                            st.error("❌ Model not available. Please check the model file.")
                        else:
                            with st.spinner("🔄 Classifying beans..."):
                                X = df[features]
                                predictions_encoded = model.predict(X)
                                predictions = [reverse_class_mapping.get(pred, f"Unknown ({pred})") 
                                             for pred in predictions_encoded]
                                
                                df['Predicted_Bean_Type'] = predictions
                                
                                try:
                                    probabilities = model.predict_proba(X)
                                    df['Confidence_%'] = (np.max(probabilities, axis=1) * 100).round(1)
                                except:
                                    pass
                                
                                st.success("✅ Classification completed!", icon="🎉")
                                
                                # Summary
                                prediction_counts = pd.Series(predictions).value_counts()
                                
                                st.markdown("### 📈 Classification Summary")
                                summary_html = '<div style="background: rgba(255,255,255,0.95); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">'
                                for bean_type, count in prediction_counts.items():
                                    emoji = bean_emojis.get(bean_type, "🫘")
                                    summary_html += f'<span style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.5rem 1rem; border-radius: 15px; margin: 0.3rem; display: inline-block;">{emoji} {bean_type}: {count}</span>'
                                summary_html += '</div>'
                                st.markdown(summary_html, unsafe_allow_html=True)
                                
                                st.markdown("**📋 Detailed Results:**")
                                st.dataframe(df, use_container_width=True)
                                
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Results",
                                    csv,
                                    "bean_classifications.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                                
                                st.balloons()
                                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 3rem; backdrop-filter: blur(10px);">
    <h3 style="color: white; margin-bottom: 1rem;">🚀 Powered by Advanced Machine Learning</h3>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; line-height: 1.6; margin: 0;">
        Our Support Vector Machine model analyzes 16 distinct features including geometric properties, 
        shape factors, and derived metrics to accurately classify 7 different bean varieties.
    </p>
    <br>
    <p style="color: rgba(255,255,255,0.7); font-style: italic; margin: 0;">
        <strong>Note:</strong> Ensure accurate feature measurements for best classification results.
    </p>
</div>
""", unsafe_allow_html=True)
