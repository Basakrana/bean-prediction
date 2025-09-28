import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Bean Classification System",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .prediction-container h2 {
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .confidence-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        display: inline-block;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .probability-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    .probability-item {
        display: flex;
        align-items: center;
        margin: 0.8rem 0;
        padding: 0.5rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #f8f9ff 0%, #ffffff 100%);
    }
    
    .bean-name {
        font-weight: 600;
        min-width: 120px;
        color: #333;
        font-size: 1rem;
    }
    
    .probability-bar {
        flex-grow: 1;
        height: 25px;
        background: #e0e0e0;
        border-radius: 12px;
        margin: 0 1rem;
        overflow: hidden;
        position: relative;
    }
    
    .probability-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        transition: width 1s ease-in-out;
    }
    
    .probability-text {
        font-weight: 600;
        color: #333;
        min-width: 60px;
        text-align: right;
    }
    
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    .section-header {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .summary-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 20px rgba(255, 154, 158, 0.3);
    }
    
    .summary-card h3 {
        margin-top: 0;
        font-size: 1.3rem;
    }
    
    .bean-count {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        display: inline-block;
        margin: 0.2rem;
        font-weight: 500;
    }
    
    .sidebar-bean {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        color: #2d3436;
        font-weight: 500;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🫘 Bean Classification System</h1>
    <p>Advanced AI-Powered Bean Type Recognition</p>
</div>
""", unsafe_allow_html=True)

# Class mapping - your manual mapping
class_mapping = {
    "DERMASON": 0,
    "SIRA": 1,
    "SEKER": 2,
    "HOROZ": 3,
    "CALI": 4,
    "BARBUNYA": 5,
    "BOMBAY": 6
}

# Reverse mapping for prediction results
reverse_class_mapping = {v: k for k, v in class_mapping.items()}

# Bean emojis for visual appeal
bean_emojis = {
    "DERMASON": "🟤",
    "SIRA": "🟡", 
    "SEKER": "⚪",
    "HOROZ": "🔴",
    "CALI": "🟢",
    "BARBUNYA": "🟣",
    "BOMBAY": "⚫"
}

# Load model
@st.cache_resource
def load_model():
    methods = [
        lambda: joblib.load('svc_best_model.pkl'),
        lambda: pickle.load(open('svc_best_model.pkl', 'rb')),
        lambda: pickle.load(open('svc_best_model.pkl', 'rb'), encoding='latin1'),
    ]
    
    for i, method in enumerate(methods):
        try:
            model = method()
            st.success(f"✅ Model loaded successfully!", icon="🎯")
            return model
        except Exception as e:
            if i == len(methods) - 1:
                st.error(f"❌ Could not load model: {str(e)}")
                return None
            continue

# Feature names
features = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

model = load_model()

if model is not None:
    # Input method selection with attractive radio buttons
    st.markdown('<div class="section-header">🔧 Choose Input Method</div>', unsafe_allow_html=True)
    input_method = st.radio("", ["🔢 Manual Input", "📁 CSV Upload"], horizontal=True, label_visibility="collapsed")
    
    if "Manual" in input_method:
        st.markdown("""
        <div class="input-section">
            <div class="section-header">📊 Enter Bean Feature Values</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create input fields in two columns
        values = []
        col1, col2 = st.columns(2)
        
        # Organize features by category
        geometric_features = features[:8]
        shape_features = features[8:]
        
        with col1:
            st.markdown("**📐 Geometric Features**")
            for feature in geometric_features:
                value = st.number_input(f"{feature}:", value=0.0, format="%.6f", key=feature)
                values.append(value)
        
        with col2:
            st.markdown("**🔍 Shape Features**")
            for feature in shape_features:
                value = st.number_input(f"{feature}:", value=0.0, format="%.6f", key=feature)
                values.append(value)
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Classify Bean", type="primary", use_container_width=True):
                try:
                    # Create DataFrame with proper column names
                    input_data = pd.DataFrame([values], columns=features)
                    
                    # Make prediction
                    prediction_encoded = model.predict(input_data)[0]
                    
                    # Convert to bean name
                    bean_name = reverse_class_mapping.get(prediction_encoded, f"Unknown ({prediction_encoded})")
                    bean_emoji = bean_emojis.get(bean_name, "🫘")
                    
                    # Display prediction with attractive styling
                    st.markdown(f"""
                    <div class="prediction-container">
                        <h2>{bean_emoji} {bean_name}</h2>
                        <p>Predicted Bean Type</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confidence and probabilities
                    try:
                        probabilities = model.predict_proba(input_data)[0]
                        max_prob = np.max(probabilities)
                        
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <span class="confidence-badge">Confidence: {max_prob:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show all predictions with attractive bars
                        st.markdown("""
                        <div class="probability-container">
                            <h3 style="text-align: center; color: #667eea; margin-bottom: 1rem;">🎯 Detailed Predictions</h3>
                        """, unsafe_allow_html=True)
                        
                        for i, prob in enumerate(probabilities):
                            bean_type = reverse_class_mapping.get(i, f"Class {i}")
                            emoji = bean_emojis.get(bean_type, "🫘")
                            percentage = prob * 100
                            
                            st.markdown(f"""
                            <div class="probability-item">
                                <div class="bean-name">{emoji} {bean_type}</div>
                                <div class="probability-bar">
                                    <div class="probability-fill" style="width: {percentage}%"></div>
                                </div>
                                <div class="probability-text">{percentage:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.info("💡 Probabilities not available for this model")
                        
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
    
    else:  # CSV Upload
        st.markdown("""
        <div class="input-section">
            <div class="section-header">📁 Batch Classification</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show expected format with style
        st.info(f"📋 **Required columns:** {', '.join(features[:4])}... (total: {len(features)} columns)")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], help="Upload your bean dataset for batch classification")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown("**📊 Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check required columns
                missing_cols = [col for col in features if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🎯 Classify All Beans", type="primary", use_container_width=True):
                            with st.spinner("🔄 Classifying beans..."):
                                # Make predictions
                                X = df[features]
                                predictions_encoded = model.predict(X)
                                
                                # Convert to bean names
                                predictions = [reverse_class_mapping.get(pred, f"Unknown ({pred})") 
                                             for pred in predictions_encoded]
                                
                                df['Predicted_Bean_Type'] = predictions
                                
                                # Add confidence scores
                                try:
                                    probabilities = model.predict_proba(X)
                                    df['Confidence_%'] = (np.max(probabilities, axis=1) * 100).round(1)
                                except:
                                    pass
                                
                                st.success("✅ Classification completed!", icon="🎉")
                                
                                # Show results summary with attractive cards
                                prediction_counts = pd.Series(predictions).value_counts()
                                
                                st.markdown("""
                                <div class="summary-card">
                                    <h3>📈 Classification Summary</h3>
                                """, unsafe_allow_html=True)
                                
                                for bean_type, count in prediction_counts.items():
                                    emoji = bean_emojis.get(bean_type, "🫘")
                                    st.markdown(f'<span class="bean-count">{emoji} {bean_type}: {count}</span>', unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Show results
                                st.markdown("**📋 Detailed Results:**")
                                st.dataframe(df, use_container_width=True)
                                
                                # Download button
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Results",
                                    csv,
                                    "bean_classifications.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

else:
    st.warning("⚠️ Model not loaded. Please ensure 'svc_best_model.pkl' is in the same directory.")

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### 🫘 Bean Types")
    st.markdown("**This system classifies 7 bean varieties:**")
    
    for bean_name, code in class_mapping.items():
        emoji = bean_emojis.get(bean_name, "🫘")
        st.markdown(f"""
        <div class="sidebar-bean">
            {emoji} {bean_name} (Code: {code})
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model info with metrics
    st.markdown("### 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Algorithm</h4>
            <p>Support Vector Machine</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Classes</h4>
            <p>7 Bean Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📏 Features Used")
    
    with st.expander("🔍 View All Features"):
        st.markdown("**Geometric Features:**")
        for feature in features[:8]:
            st.write(f"• {feature}")
        
        st.markdown("**Shape Features:**")
        for feature in features[8:]:
            st.write(f"• {feature}")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h4>🚀 Powered by AI</h4>
        <p style="margin: 0;">Advanced machine learning for precise bean classification</p>
    </div>
    """, unsafe_allow_html=True)