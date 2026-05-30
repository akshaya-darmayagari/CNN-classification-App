# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# 1. Page Config
st.set_page_config(
    page_title="CNN classification app",
    page_icon="🎨",
    layout="wide"
)

# 2. Enhanced CSS for Vivid Colors, Text Gradients, and Soft Badges
st.markdown("""
    <style>
        /* Light base background */
        .stApp {
            background-color: #f5f7fb;
        }
        
        /* Gradient Text for the Main Heading */
        .gradient-heading {
            font-size: 3.2rem;
            font-weight: 900;
            background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            padding-bottom: 10px;
        }
        
        .title-divider {
            height: 4px;
            background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
            border-radius: 2px;
            margin-bottom: 30px;
        }
        
        /* Vivid Section Subheaders */
        .section-header-left {
            color: #4f46e5; /* Electric Indigo */
            font-weight: 800;
            font-size: 1.5rem;
            margin-bottom: 15px;
            border-left: 5px solid #4f46e5;
            padding-left: 10px;
        }
        
        .section-header-right {
            color: #db2777; /* Deep Magenta */
            font-weight: 800;
            font-size: 1.5rem;
            margin-bottom: 15px;
            border-left: 5px solid #db2777;
            padding-left: 10px;
        }

        /* Colorful Dashboard Metric Card */
        .prediction-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
            border: 2px solid #e0e7ff;
            text-align: center;
            margin-bottom: 25px;
        }
        .prediction-label {
            font-size: 0.9rem;
            color: #6b7280;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .prediction-class {
            font-size: 2.6rem;
            font-weight: 900;
            /* Bright Violet/Indigo Text for the Winner */
            color: #6366f1; 
            margin: 10px 0;
            text-shadow: 1px 1px 2px rgba(99, 102, 241, 0.1);
        }
        .prediction-confidence {
            font-size: 1.25rem;
            color: #10b981; /* Success Green */
            font-weight: 700;
        }

        /* Colorful Sidebar Badge Styling */
        .sidebar-badge {
            background-color: #e0e7ff;
            color: #4338ca;
            padding: 8px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 8px;
            display: inline-block;
            width: 100%;
            border-left: 4px solid #4f46e5;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Class Mapping with Emojis
CLASS_MAPPING = {
    'airplane': '✈️ Airplane',
    'automobile': '🚗 Automobile',
    'bird': '🐦 Bird',
    'cat': '🐱 Cat',
    'deer': '🦌 Deer',
    'dog': '🐶 Dog',
    'frog': '🐸 Frog',
    'horse': '🐴 Horse',
    'ship': '🚢 Ship',
    'truck': '🚛 Truck'
}
CLASS_NAMES = list(CLASS_MAPPING.keys())

# Sample online images
SAMPLES = {
    "Select a sample image...": None,
    "Airplane (Fighter Jet)": "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=300&auto=format&fit=crop&q=80",
    "Automobile (Sports Car)": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=300&auto=format&fit=crop&q=80",
    "Cat (Tabby Cat)": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300&auto=format&fit=crop&q=80",
    "Dog (Golden Retriever)": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=300&auto=format&fit=crop&q=80"
}

# 4. Model Loader
@st.cache_resource
def load_saved_model():
    try:
        return tf.keras.models.load_model('cifar10_model.h5')
    except Exception:
        return None

def preprocess(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

model = load_saved_model()

# 5. Colorful Sidebar containing Badge Elements
with st.sidebar:
    st.markdown("<h2 style='color: #4f46e5; font-weight:800;'>App Navigation</h2>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='color: #4b5563; font-weight: 600;'>Supported Classes:</p>", unsafe_allow_html=True)
    for category in CLASS_MAPPING.values():
        st.markdown(f'<div class="sidebar-badge">{category}</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Framework: Keras & TensorFlow")

# 6. Vibrant Main Header
st.markdown('<h1 class="gradient-heading">CNN classification app</h1>', unsafe_allow_html=True)
st.markdown('<div class="title-divider"></div>', unsafe_allow_html=True)

if model is None:
    st.error("🚨 **Model missing.** Could not locate `cifar10_model.h5` in the root directory. Please verify that the training process has run successfully.")
else:
    # App Grid System
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header-left">📥 Input Source</div>', unsafe_allow_html=True)
        
        # Interactive Sample Selector
        sample_choice = st.selectbox(
            "Option A: Test with a preloaded sample image", 
            options=list(SAMPLES.keys())
        )
        
        # File Uploader
        uploaded_file = st.file_uploader(
            "Option B: Upload your own image file", 
            type=["png", "jpg", "jpeg"]
        )
        
        active_img = None
        
        if uploaded_file is not None:
            active_img = Image.open(uploaded_file)
        elif sample_choice != "Select a sample image..." and SAMPLES[sample_choice] is not None:
            with st.spinner("Downloading sample image..."):
                try:
                    response = requests.get(SAMPLES[sample_choice])
                    active_img = Image.open(BytesIO(response.content))
                except Exception as e:
                    st.error(f"Error fetching sample image: {e}")
        
        if active_img is not None:
            st.image(active_img, caption="Active Image Preview", use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header-right">📊 Classification Output</div>', unsafe_allow_html=True)
        
        if active_img is None:
            st.markdown("""
                <div style="background-color: #ffffff; border: 2px dashed #cbd5e0; border-radius: 12px; padding: 40px; text-align: center; margin-top: 10px;">
                    <h4 style="color: #4b5563; margin-bottom: 5px; font-weight: 700;">Awaiting Input Image</h4>
                    <p style="color: #9ca3af; font-size: 0.95rem; margin: 0;">Please select a sample or upload a file on the left to display metrics.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing image..."):
                tensor = preprocess(active_img)
                preds = model.predict(tensor)[0]
                
                best_match_idx = np.argmax(preds)
                label_name = CLASS_NAMES[best_match_idx]
                display_label = CLASS_MAPPING[label_name]
                confidence_score = preds[best_match_idx] * 100
                
                # Main Prediction Card displaying vibrant Violet text
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-label">Top Predicted Category</div>
                        <div class="prediction-class">{display_label.upper()}</div>
                        <div class="prediction-confidence">Confidence Match: {confidence_score:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Visual breakdowns with color indicator labels
                st.markdown("<h4 style='color: #1f2937; font-weight: 700;'>Probability Breakdown</h4>", unsafe_allow_html=True)
                for i, score in enumerate(preds):
                    class_tag = CLASS_MAPPING[CLASS_NAMES[i]]
                    col_txt, col_pb = st.columns([2, 5])
                    with col_txt:
                        # Highlight active prediction text with bright blue
                        if i == best_match_idx:
                            st.markdown(f"<span style='color: #4f46e5; font-weight: 800;'>{class_tag}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color: #4b5563; font-weight: 500;'>{class_tag}</span>", unsafe_allow_html=True)
                    with col_pb:
                        if i == best_match_idx:
                            st.progress(float(score), text=f"Match — {score*100:.1f}%")
                        else:
                            st.progress(float(score), text=f"{score*100:.1f}%")