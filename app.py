"""
Magic Story Maker - turns an image into a kid-friendly audio story.

ISOM5240 Individual Assignment.
"""

import io
import random
import re

import streamlit as st
from PIL import Image
from gtts import gTTS
from transformers import pipeline


# ---------- Page config ----------
st.set_page_config(
    page_title="Magic Story Maker",
    page_icon="🦄",
    layout="centered",
)


# ---------- Constants ----------

# Cute rotating messages shown while each pipeline stage is running.
WAITING_MESSAGES = {
    "caption": [
        "Teddy bear is looking at your picture...",
        "Owl is using its big round eyes to see...",
        "Bunny is hopping around your picture...",
        "Curious kitten is having a peek...",
        "Friendly fox is studying your picture...",
    ],
    "story": [
        "The story fairy is sprinkling magic dust...",
        "Unicorn is whispering ideas in my ear...",
        "Mixing rainbow colors of imagination...",
        "Friendly dragon is breathing creative fire...",
        "Wizard is brewing up a wonderful tale...",
        "Painting your story with sparkly stars...",
    ],
    "audio": [
        "Mr. Frog is warming up his voice...",
        "Songbird is practicing the words...",
        "Recording in the magic sound studio...",
        "Tuning the storybook microphone...",
    ],
}

STAGE_EMOJI = {"caption": "🐻", "story": "🦄", "audio": "🎤"}

# Words that should not appear in a story for young children.
UNSAFE_KEYWORDS = [
    "kill", "killed", "killing", "murder", "murdered",
    "blood", "bloody", "gun", "shoot", "shot", "weapon",
    "knife", "stab", "stabbed", "die", "died", "dead",
    "death", "corpse",
    "demon", "devil", "satan", "hell", "evil",
    "horror", "terror", "nightmare", "torture",
    "sex", "sexual", "naked", "nude", "porn",
    "drug", "drugs", "cocaine", "heroin", "drunk", "alcohol",
    "suicide", "suicidal",
    "racist", "racism",
]

# Used if every generation attempt fails the safety check.
FALLBACK_STORY = (
    "In a sunny little meadow full of wildflowers, a kind friend was having "
    "a wonderful day. They skipped through the soft grass and sang cheerful "
    "songs to the butterflies. A fluffy bunny hopped over to say hello, and "
    "they shared a basket of sweet berries together. The sky turned pink "
    "and gold as the evening came, and the stars began to twinkle overhead. "
    "They smiled and waved goodnight, knowing tomorrow would bring even more "
    "fun adventures."
)


# ---------- Styling ----------

def inject_css():
    """Apply a colorful, kid-friendly look on top of Streamlit's defaults."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Bubblegum+Sans&display=swap');

        .stApp {
            background: linear-gradient(135deg, #ffeef8 0%, #fff5e6 25%,
                #f0fff4 50%, #e6f7ff 75%, #f3e8ff 100%);
        }
        h1 {
            font-family: 'Bubblegum Sans', cursive !important;
            text-align: center;
            color: #ff6fb5 !important;
            font-size: 3rem !important;
            text-shadow: 3px 3px 0px #fff5b1, 6px 6px 0px rgba(255, 111, 181, 0.2);
            padding: 1rem 0;
        }
        h3 {
            font-family: 'Bubblegum Sans', cursive !important;
            color: #7c5fa8 !important;
        }
        .stMarkdown, p, label {
            font-family: 'Fredoka', sans-serif !important;
            font-size: 1.1rem !important;
        }
        .stButton > button, .stDownloadButton > button {
            font-family: 'Fredoka', sans-serif !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #ff9ec7 0%, #ffd96b 100%) !important;
            color: white !important;
            border: 4px solid white !important;
            border-radius: 50px !important;
            padding: 0.8rem 2rem !important;
            box-shadow: 0 4px 15px rgba(255, 158, 199, 0.4) !important;
            transition: transform 0.2s !important;
            width: 100% !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 6px 20px rgba(255, 158, 199, 0.6) !important;
        }
        [data-testid="stFileUploader"] {
            background: white;
            border-radius: 25px;
            padding: 1.5rem;
            border: 4px dashed #ff9ec7;
        }
        .story-box {
            background: white;
            border-radius: 25px;
            padding: 2rem;
            margin: 1rem 0;
            border: 4px solid #ffd96b;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            font-family: 'Fredoka', sans-serif;
            font-size: 1.25rem;
            line-height: 1.8;
            color: #4a3f55;
        }
        .waiting-message {
            text-align: center;
            font-family: 'Bubblegum Sans', cursive;
            font-size: 1.6rem;
            color: #9f7aea;
            padding: 1.5rem;
            animation: bounce 1s infinite alternate;
        }
        @keyframes bounce {
            from { transform: translateY(0); }
            to   { transform: translateY(-10px); }
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Helpers ----------

def is_safe_for_kids(text):
    """Return True if no blacklisted word appears in the text as a whole word."""
    text_lower = text.lower()
    for word in UNSAFE_KEYWORDS:
        # \b word-boundaries avoid false hits like "kill" inside "skill".
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
            return False
    return True


def trim_to_word_range(text, min_words=50, max_words=100):
    """Force the story length into the [min_words, max_words] range.

    If too long, truncate at the last sentence boundary inside the window.
    If too short, append friendly closing sentences until it's long enough.
    """
    words = text.split()

    # Too long: trim back to a clean sentence ending.
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        cut_idx = -1
        for punct in [".", "!", "?"]:
            idx = truncated.rfind(punct)
            if idx > len(truncated) * 0.6 and idx > cut_idx:
                cut_idx = idx
        if cut_idx > 0:
            truncated = truncated[: cut_idx + 1]
        else:
            truncated = truncated.rstrip(",;: ") + "."
        text = truncated
        words = text.split()

    # Too short: stack friendly sentences until we hit the minimum.
    if len(words) < min_words:
        padding_sentences = [
            "They smiled and laughed together under the warm sunshine.",
            "Birds sang sweet songs and butterflies danced all around them.",
            "They shared yummy snacks with their new friends in the meadow.",
            "Everyone felt cozy and joyful as the day went on.",
            "It was the start of many more wonderful adventures.",
        ]
        text = text.rstrip(".!? ") + "."
        for sentence in padding_sentences:
            text = text + " " + sentence
            if len(text.split()) >= min_words:
                break

    return text


def build_prompt(caption):
    """Build the instruction prompt fed to the story model."""
    return (
        f"Write a short, happy bedtime story for a young child (age 5) about: "
        f"{caption}. Make the story gentle, magical, and friendly with a happy "
        f"ending. About 80 words."
    )


# ---------- Models (cached so they only load once per session) ----------

@st.cache_resource(show_spinner=False)
def load_caption_model():
    """Load the BLIP image-captioning pipeline."""
    return pipeline(
        task="image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )


@st.cache_resource(show_spinner=False)
def load_story_model():
    """Load the Flan-T5 text-generation pipeline."""
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
    )


# ---------- Pipeline ----------

def caption_image(model, image):
    """Generate a one-line description of the image."""
    return model(image)[0]["generated_text"].strip()


def make_story(model, caption, max_attempts=3):
    """Generate a kid-friendly story (50-100 words) from an image caption.

    Retries up to max_attempts times if the safety check fails.
    Falls back to a fixed safe story if every attempt fails.
    """
    prompt = build_prompt(caption)

    for _ in range(max_attempts):
        result = model(
            prompt,
            max_new_tokens=180,
            min_new_tokens=80,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        story = result[0]["generated_text"].strip()
        story = trim_to_word_range(story)

        if is_safe_for_kids(story):
            return story

    return FALLBACK_STORY


def make_audio(text):
    """Convert a story to MP3 audio bytes via gTTS."""
    # Drop emojis and other non-ASCII so gTTS doesn't try to read them aloud.
    clean = "".join(ch for ch in text if ch.isascii() or ch in " .,!?'\"\n")

    tts = gTTS(text=clean, lang="en", tld="com", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ---------- UI helpers ----------

def show_waiting(placeholder, stage):
    """Display a randomly chosen cute waiting message for the given stage."""
    msg = random.choice(WAITING_MESSAGES[stage])
    placeholder.markdown(
        f"<div class='waiting-message'>{STAGE_EMOJI[stage]} {msg}</div>",
        unsafe_allow_html=True,
    )


def run_pipeline(caption_model, story_model):
    """Run caption -> story -> audio and stash the results in session_state."""
    placeholder = st.empty()

    try:
        show_waiting(placeholder, "caption")
        caption = caption_image(caption_model, st.session_state.image)

        show_waiting(placeholder, "story")
        story = make_story(story_model, caption)
        st.session_state.story = story

        show_waiting(placeholder, "audio")
        st.session_state.audio = make_audio(story)

        placeholder.empty()
    except Exception as exc:
        placeholder.empty()
        st.error("Oops! 🙈 Something went wiggly. Let's try another picture!")
        print(f"[ERROR] {exc}")


# ---------- Main ----------

def main():
    """Top-level Streamlit page."""
    inject_css()

    st.markdown("# 🦄 Magic Story Maker 🌈")
    st.markdown(
        "<p style='text-align:center; font-size:1.3rem; color:#7c5fa8;'>"
        "📸 Show me a picture, and I'll tell you a magical story! ✨"
        "</p>",
        unsafe_allow_html=True,
    )

    # Initialize session state on first load.
    for key in ("image", "story", "audio"):
        if key not in st.session_state:
            st.session_state[key] = None

    # Load both models up front. They're cached, so this only runs once.
    with st.spinner("🌟 Waking up the story magic... (the first time can take a minute) 🌟"):
        caption_model = load_caption_model()
        story_model = load_story_model()

    # Step 1: image upload.
    st.markdown("### 1️⃣ Pick a picture! 📸")
    uploaded = st.file_uploader(
        "Choose a picture",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        # Convert to RGB so PNGs with transparency don't break BLIP.
        image = Image.open(uploaded).convert("RGB")
        st.session_state.image = image
        st.image(image, caption="Your picture! 🖼️", use_container_width=True)

    # Step 2: generate.
    if st.session_state.image is not None:
        st.markdown("### 2️⃣ Make my story! ✨")
        if st.button("✨ Tell me a story! ✨", key="generate_btn"):
            # Clear previous results so the UI updates cleanly.
            st.session_state.story = None
            st.session_state.audio = None
            run_pipeline(caption_model, story_model)

    # Step 3 & 4: show the story and the audio.
    if st.session_state.story:
        st.markdown("### 3️⃣ Story time! 📖")
        st.markdown(
            f"<div class='story-box'>📖 <b>Your Story:</b><br><br>"
            f"{st.session_state.story}</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.audio:
            st.markdown("### 4️⃣ Listen along! 🎧")
            st.audio(st.session_state.audio, format="audio/mp3")
            st.download_button(
                label="💾 Save my story",
                data=st.session_state.audio,
                file_name="my_magic_story.mp3",
                mime="audio/mp3",
            )

        # Replay button: generates a fresh story from the same picture.
        st.markdown("---")
        if st.button("🎁 Tell me a different story!", key="retry_btn"):
            st.session_state.story = None
            st.session_state.audio = None
            run_pipeline(caption_model, story_model)


if __name__ == "__main__":
    main()
