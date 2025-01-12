import os
import openai
import streamlit as st

st.title("Zero-shot Sentiment Analysis with OpenAI's")

# Ask the user for their OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Initialize the OpenAI client only if the API key is provided
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    openai.api_key = os.getenv('OPENAI_API_KEY')

    from openai import OpenAI
    client = OpenAI()

    def gpt_classify_sentiment(prompt, emotions):
        system_prompt = f'''
        You are an emotionally intelligent assistant.
        Classify the sentiment of the user's text with ONLY ONE OF THE FOLLOWING EMOTIONS: {emotions}.
        After classifying the text, respond with the emotion ONLY.
        '''
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=20,
            temperature=0
        )
        r = response.choices[0].message.content
        if not r:
            r = 'N/A'
        return r

    with st.form(key='my_form'):
        default_emotion = 'positive,negative,neutral'
        emotions = st.text_input('Emotions:', value=default_emotion)
        text = st.text_area(label='Text to classify:')
        submit_button = st.form_submit_button(label='Classify Sentiment')

        if submit_button and text:
            emotion = gpt_classify_sentiment(text, emotions)
            result = f'{text}  => {emotion}'
            st.write(result)
else:
    st.warning("Please enter your OpenAI API Key to proceed.")
