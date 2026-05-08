# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_text = ""   # to be completed
    return story_text

# text2audio
def text2audio(story_text):
    audio_data = ""     # to be completed
    return audio_data

# main part
# ... to be completed
