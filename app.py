import streamlit as st
import os
from PIL import Image
from huggingface_hub import InferenceClient, HfApi
from huggingface_hub.errors import HfHubHTTPError

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="HF Universal Tester", page_icon="🤗", layout="centered")

# Securely load the token from Streamlit Secrets or Environment Variables
hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")

if not hf_token:
    st.error("⚠️ Hugging Face Token not found. Please set HF_TOKEN in .streamlit/secrets.toml")
    st.stop()

# Initialize Clients
# InferenceClient runs the models, HfApi fetches model metadata
client = InferenceClient(api_key=hf_token)
api = HfApi(token=hf_token)

# --- 2. SIDEBAR: MODEL SELECTION ---
with st.sidebar:
    st.title("🤗 HF Tester")
    st.markdown("Test serverless models dynamically by their Hub ID.")
    model_id = st.text_input("Enter Model ID:", placeholder="e.g., google/gemma-2b")
    
    st.markdown("---")
    st.markdown("### Examples to try:")
    st.code("google/gemma-2b", language="text")
    st.code("black-forest-labs/FLUX.1-schnell", language="text")
    st.code("google/vit-base-patch16-224", language="text")
    st.code("facebook/bart-large-cnn", language="text")

# --- 3. MAIN APP LOGIC ---
if not model_id:
    st.info("👈 Enter a Hugging Face Model ID in the sidebar to get started.")
    st.stop()

st.header(f"Testing: `{model_id}`")

# Fetch Model Metadata to determine the UI
try:
    with st.spinner("Fetching model metadata..."):
        model_info = api.model_info(model_id)
        task = model_info.pipeline_tag
        
    if not task:
        st.warning("Could not detect a specific task (pipeline_tag) for this model. The Serverless API might not support it.")
        st.stop()
        
    st.caption(f"**Detected Task:** `{task}` | **Total Hub Downloads:** {model_info.downloads}")

except HfHubHTTPError as e:
    if e.response.status_code == 404:
        st.error(f"Model `{model_id}` not found. Check for typos!")
    elif e.response.status_code == 401:
        st.error("Unauthorized. Check if your HF_TOKEN is valid.")
    else:
        st.error(f"API Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error fetching model data: {e}")
    st.stop()

st.divider()

# --- 4. DYNAMIC UI ROUTING BASED ON TASK ---

# Modality 1: Text Generation
if task in ["text-generation", "text2text-generation"]:
    prompt = st.text_area("Enter your prompt:", height=150)
    if st.button("Generate Text"):
        with st.spinner("Generating (may take up to 60s if the model is waking up)..."):
            try:
                result = client.text_generation(prompt, model=model_id, max_new_tokens=250)
                st.success("Success!")
                st.write(result)
            except Exception as e:
                st.error(f"Inference Error: {e}")

# Modality 2: Text-to-Image
elif task == "text-to-image":
    prompt = st.text_area("Describe the image you want to generate:")
    if st.button("Generate Image"):
        with st.spinner("Painting (may take up to 60s if the model is waking up)..."):
            try:
                # The text_to_image method returns a PIL Image object directly
                image = client.text_to_image(prompt, model=model_id)
                st.image(image, caption=prompt, use_container_width=True)
            except Exception as e:
                st.error(f"Inference Error: {e}")

# Modality 3: Image Classification
elif task == "image-classification":
    uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])
    if uploaded_file and st.button("Classify Image"):
        with st.spinner("Classifying..."):
            try:
                img = Image.open(uploaded_file)
                st.image(img, width=300)
                result = client.image_classification(img, model=model_id)
                st.success("Success!")
                st.json(result) # Output the raw JSON classification scores
            except Exception as e:
                st.error(f"Inference Error: {e}")

# Modality 4: Summarization
elif task == "summarization":
    prompt = st.text_area("Paste text to summarize:", height=200)
    if st.button("Summarize"):
        with st.spinner("Summarizing text..."):
            try:
                result = client.summarization(prompt, model=model_id)
                st.success("Success!")
                st.write(result)
            except Exception as e:
                st.error(f"Inference Error: {e}")

# Fallback for unsupported tasks
else:
    st.warning(f"UI support for the `{task}` task isn't built into this Streamlit app yet!")
    st.info("You can easily add it by adding a new `elif task == '...'` block and using the corresponding method from `huggingface_hub.InferenceClient`.")
