# Program title: Magic Picture Storyteller
# This Streamlit app turns an uploaded image into a short child-friendly story with audio.

import io
import re
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline


CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
STORY_MODEL_NAME = "google/flan-t5-base"
TTS_MODEL_NAME = "facebook/mms-tts-eng"

MIN_STORY_WORDS = 50
MAX_STORY_WORDS = 100

UNSAFE_WORDS = {
    "blood", "bloody", "kill", "killed", "killing", "murder", "murdered",
    "gun", "guns", "weapon", "weapons", "knife", "knives", "bomb",
    "death", "dead", "die", "dying", "suicide", "horror", "monster",
    "scary", "terrifying", "terror", "war", "violent", "violence",
    "sex", "sexy", "nude", "naked", "drugs", "alcohol"
}


@st.cache_resource
def load_caption_model():
    """Load the image captioning model."""
    return pipeline("image-to-text", model=CAPTION_MODEL_NAME)


@st.cache_resource
def load_story_model():
    """Load the story generation model."""
    return pipeline("text2text-generation", model=STORY_MODEL_NAME)


@st.cache_resource
def load_tts_model():
    """Load the text-to-speech model."""
    return pipeline("text-to-speech", model=TTS_MODEL_NAME)


def count_words(text: str) -> int:
    """Count words in a text string."""
    words = re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", text)
    return len(words)


def clean_text(text: str) -> str:
    """Clean extra spaces and simple unwanted prefixes."""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(Story:|Children's story:|Here is a story:)\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def contains_unsafe_content(text: str) -> bool:
    """Check whether the generated text contains unsafe words."""
    lowered_text = text.lower()
    for word in UNSAFE_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered_text):
            return True
    return False


def make_safe_caption(caption: str) -> str:
    """Replace unsafe captions with a general child-friendly scene."""
    if contains_unsafe_content(caption):
        return "a bright and friendly scene"
    return caption


def trim_story_to_limit(story: str, max_words: int = MAX_STORY_WORDS) -> str:
    """Trim a story to the maximum word limit while keeping sentence endings when possible."""
    words = story.split()

    if len(words) <= max_words:
        return story

    trimmed = " ".join(words[:max_words])
    last_period = trimmed.rfind(".")
    last_exclamation = trimmed.rfind("!")
    last_question = trimmed.rfind("?")
    last_sentence_end = max(last_period, last_exclamation, last_question)

    if last_sentence_end > 40:
        trimmed = trimmed[:last_sentence_end + 1]

    return trimmed.strip()


def build_fallback_story(caption: str) -> str:
    """Create a safe fallback story if the model output is not suitable."""
    safe_caption = make_safe_caption(caption)

    story = (
        f"One sunny day, a little friend noticed {safe_caption}. "
        "It felt like the start of a tiny adventure. "
        "The friend looked around, smiled, and decided to do something kind. "
        "Soon, everyone nearby felt happier and more hopeful. "
        "By the end of the day, the little friend learned that kindness can turn "
        "an ordinary moment into a wonderful story."
    )

    return trim_story_to_limit(clean_text(story))


def generate_caption(image: Image.Image) -> str:
    """Generate a caption from the uploaded image."""
    caption_model = load_caption_model()
    result = caption_model(image)

    if not result or "generated_text" not in result[0]:
        return "a bright and friendly scene"

    caption = clean_text(result[0]["generated_text"])
    return make_safe_caption(caption)


def generate_story(caption: str) -> str:
    """Generate a short child-friendly story based on an image caption."""
    story_model = load_story_model()
    safe_caption = make_safe_caption(caption)

    prompt = (
        "Write a warm and simple story for children aged 3 to 10. "
        "The story must be based on this image description: "
        f"{safe_caption}. "
        "Use 50 to 100 words. "
        "Use simple English. "
        "Make the story cheerful, kind, and easy to understand. "
        "The story must have a happy ending. "
        "Do not include violence, fear, romance, adult topics, unsafe behavior, or scary content."
    )

    try:
        result = story_model(
            prompt,
            max_new_tokens=140,
            min_new_tokens=70,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )

        story = clean_text(result[0]["generated_text"])
        story = trim_story_to_limit(story)

        if contains_unsafe_content(story):
            return build_fallback_story(safe_caption)

        if count_words(story) < MIN_STORY_WORDS or count_words(story) > MAX_STORY_WORDS:
            return build_fallback_story(safe_caption)

        return story

    except Exception:
        return build_fallback_story(safe_caption)


def generate_audio(story: str) -> Tuple[np.ndarray, int]:
    """Convert the story text into speech audio."""
    tts_model = load_tts_model()
    speech_output = tts_model(story)

    audio_array = speech_output["audio"]
    sample_rate = speech_output["sampling_rate"]

    return audio_array, sample_rate


def show_welcome_message():
    """Display the main page title and instructions."""
    st.set_page_config(
        page_title="Magic Picture Storyteller",
        page_icon="🌈",
        layout="centered"
    )

    st.title("🌈 Magic Picture Storyteller")
    st.subheader("Turn your picture into a happy little story!")

    st.write(
        "Upload a picture, and the story fairy will create a short story "
        "and read it aloud for you."
    )


def initialize_session_state():
    """Initialize Streamlit session state values."""
    if "image_id" not in st.session_state:
        st.session_state.image_id = None

    if "caption" not in st.session_state:
        st.session_state.caption = None

    if "story" not in st.session_state:
        st.session_state.story = None

    if "audio_array" not in st.session_state:
        st.session_state.audio_array = None

    if "sample_rate" not in st.session_state:
        st.session_state.sample_rate = None


def reset_story_state_if_new_image(image_bytes: bytes):
    """Reset generated results when a new image is uploaded."""
    current_image_id = hash(image_bytes)

    if st.session_state.image_id != current_image_id:
        st.session_state.image_id = current_image_id
        st.session_state.caption = None
        st.session_state.story = None
        st.session_state.audio_array = None
        st.session_state.sample_rate = None


def main():
    """Run the Streamlit application."""
    show_welcome_message()
    initialize_session_state()

    uploaded_file = st.file_uploader(
        "📸 Choose a picture",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Pick a picture when you are ready. The magic story will start here! ✨")
        return

    image_bytes = uploaded_file.getvalue()
    reset_story_state_if_new_image(image_bytes)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    st.image(image, caption="Your picture", use_container_width=True)

    make_story_button = st.button("✨ Make my story", type="primary")

    if make_story_button:
        with st.spinner("🐰 The little bunny detective is looking at your picture..."):
            caption = generate_caption(image)
            st.session_state.caption = caption

        with st.spinner("✨ The story fairy is writing a happy little tale..."):
            story = generate_story(caption)
            st.session_state.story = story

        with st.spinner("🦜 The friendly parrot is practicing the story aloud..."):
            audio_array, sample_rate = generate_audio(story)
            st.session_state.audio_array = audio_array
            st.session_state.sample_rate = sample_rate

        st.success("🎉 Your story is ready!")

    if st.session_state.story:
        with st.expander("🔍 What did the app see in the picture?"):
            st.write(st.session_state.caption)

        st.markdown("### 📖 Your Happy Story")
        st.write(st.session_state.story)

    if st.session_state.audio_array is not None and st.session_state.sample_rate is not None:
        st.markdown("### 🔊 Listen to the Story")
        st.audio(
            st.session_state.audio_array,
            sample_rate=st.session_state.sample_rate
        )

    st.markdown("---")
    st.caption("Made for young storytellers aged 3 to 10.")


if __name__ == "__main__":
    main()
