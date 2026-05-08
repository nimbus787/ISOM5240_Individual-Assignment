import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline
)

@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def img2text(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")

    processor, model = load_caption_model()
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    text = processor.decode(output[0], skip_special_tokens=True)
    return text

@st.cache_resource
def load_story_pipe():
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

@st.cache_resource
def load_audio_pipe():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")


st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
st.header("Turn Your Image to Audio Story")

uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    st.text("Processing img2text...")
    scenario = img2text(uploaded_file)
    st.write(f"**Scenario:** {scenario}")

    st.text("Generating a story...")
    story_pipe = load_story_pipe()
    story_results = story_pipe(
        scenario,
        max_new_tokens=150,
        do_sample=True
    )
    story = story_results[0]["generated_text"]
    st.write(f"**Story:** {story}")

    st.text("Generating audio data...")
    audio_pipe = load_audio_pipe()
    audio_data = audio_pipe(story)

    st.audio(
        audio_data["audio"],
        sample_rate=audio_data["sampling_rate"]
    )
