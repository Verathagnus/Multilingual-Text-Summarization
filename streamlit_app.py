import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Concatenate, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

import pickle
import streamlit as st
from ftlangdetect import detect
import iso639
import streamlit.components.v1 as components
import os
gpt2_tokenizer = None
gpt2_model = None
from transformers import (
    # GPT2Config,
    #                       GPT2Tokenizer,
    #                       GPT2Model,
                          BertTokenizer, 
                          BertModel)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
class_names = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
import os
gpt2_tokenizer = None
gpt2_model = None
# gpt2_model = GPT2Model.from_pretrained("gpt2")
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# gpt2_tokenizer.padding_side = "left"
# gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
# Define preprocessing function with smaller max length
def tokenize_sample(texts, tokenizer="bert"):
    if tokenizer == "gpt2":
        return gpt2_tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt', max_length=128)
    return bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
def get_embeddings(text, model_type="bert"):
    tokenized_text = tokenize_sample(text, model_type)
    if model_type =="gpt2":
        outputs = gpt2_model(**tokenized_text)
    else:
        outputs = bert_model(**tokenized_text)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()  # Get the embeddings for [CLS] token
    return embeddings

# path_to_models = "."
path_to_models = os.environ['RAILWAY_VOLUME_MOUNT_PATH']+"/storage"
emotion_classifier_map={
"Naive Bayes":f"{path_to_models}/models/naive_bayes_model.sav",
"Logistic Regression":f"{path_to_models}/models/logistic_regression_model.sav",
"KNN":f"{path_to_models}/models/knn_model.sav",
"KMeans":f"{path_to_models}/models/kmeans_model.sav",
"SVM":f"{path_to_models}/models/svm_model.sav",
"Decision Tree":f"{path_to_models}/models/decision_tree_model.sav",
"Random Forest":f"{path_to_models}/models/random_forest_model.sav"
}
summarizer_map={
    "Bengali":f"{path_to_models}/models/bengali_summarization_model.sav",
}
# print(os.listdir())
# print(os.environ["RAILWAY_VOLUME_MOUNT_PATH"])
# print(os.listdir(os.environ["RAILWAY_VOLUME_MOUNT_PATH"]+"/storage"))
summarizer_models=dict()
for i in summarizer_map:
    with open(summarizer_map[i], 'rb') as file:
        summarizer_models[i] = pickle.load(file)
emotion_classfier_models=dict()
for i in emotion_classifier_map:
    with open(emotion_classifier_map[i], 'rb') as file:
        emotion_classfier_models[i] = pickle.load(file)
def get_emotion_prediction(input, model_name):
    if model_name in emotion_classfier_models:
        return class_names[emotion_classfier_models[model_name].predict(get_embeddings(input))[0]]
    else:
        raise ValueError("Model type should be of the types: {}".format(", ".join(list(emotion_classfier_models.keys()))))
    
def decode_sequence(input_seq, max_summary_len, encoder_model, decoder_model, target_word_index, reverse_target_word_index):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def summarize_text(text, x_tokenizer, max_text_len, max_summary_len, encoder_model, decoder_model, target_word_index, reverse_target_word_index):
    tokenized_sentence = pad_sequences(x_tokenizer.texts_to_sequences([text]),  maxlen=max_text_len, padding='post')[0]
    return decode_sequence(tokenized_sentence.reshape(1,max_text_len), max_summary_len, encoder_model, decoder_model, target_word_index, reverse_target_word_index)

def main():
    list_of_tabs = st.tabs(["Indic Multilingual Text Summarization", "Indic Multilingual Emotion Detection"])
    # Title of the web app
    with list_of_tabs[0]:
        st.title('Indic Multilingual Text Summarization')
        # print(os.listdir())
        # print(os.environ["RAILWAY_VOLUME_MOUNT_PATH"])
        # print(os.listdir(os.environ["RAILWAY_VOLUME_MOUNT_PATH"]))
        
        # Input text from the user
        input_sentence_emotion = st.text_input('Enter paragraph/text', key="summarize")

        # Model selection
        # model_option = st.selectbox('Select the model', list(models.keys()))
        # Result initialization
        result = None
        error = None
        langlist = {"bn": "Bengali"}
        # Prediction button
        if st.button('Summarize'):
            lang = detect(text=input_sentence_emotion, low_memory=False)['lang']
            if lang in langlist:
                result = summarize_text(input_sentence_emotion, summarizer_models[langlist[lang]]["x_tokenizer"], summarizer_models[langlist[lang]]["max_text_len"],summarizer_models[langlist[lang]]['max_summary_len'], summarizer_models[langlist[lang]]['encoder_model'], summarizer_models[langlist[lang]]['decoder_model'], summarizer_models[langlist[lang]]['target_word_index'], summarizer_models[langlist[lang]]['reverse_target_word_index']).replace("start ", "").replace(" end", "")
            else:
                error = f"{iso639.Language.from_part1(lang).name} is not supported.\n List of supported languages: {', '.join(langlist.values())}"
        st.markdown(f"Current language support: Bengali")
        # Display the result
        if result:
            st.success(f'Summary: {result}')
        if error:
            st.error(f'Error: {error}')
        # Credits
        # Credits
        
    
    with list_of_tabs[1]:
        st.title('Indic Multilingual Emotion Detection')
        # print(os.listdir())
        # print(os.environ["RAILWAY_VOLUME_MOUNT_PATH"])
        # print(os.listdir(os.environ["RAILWAY_VOLUME_MOUNT_PATH"]))
        
        # Input text from the user
        input_sentence_emotion = st.text_input('Enter a sentence', key="emotion")

        # Model selection
        model_option = st.selectbox('Select the model', list(emotion_classfier_models.keys()))

        # Result initialization
        result = None
        error = None
        langlist = {"hi": "Hindi"}
        # Prediction button
        if st.button('Predict Emotion'):
            lang = detect(text=input_sentence_emotion, low_memory=False)['lang']
            if lang in langlist:
                result = get_emotion_prediction(input_sentence_emotion, model_option)
            else:
                error = f"{iso639.Language.from_part1(lang).name} is not supported.\n List of supported languages: {', '.join(langlist.values())}"
        st.markdown(f"Current language support: Hindi")
        # Display the result
        if result:
            st.success(f'Prediction: {result}')
        if error:
            st.error(f'Error: {error}')
        # Credits
        # Credits
    st.markdown("---")  # Separator
    st.markdown("""## Contributors  
- Mr. Bishwaraj Paul  
**Role:** Intern   
**Email:** bishwaraj.paul98@gmail.com / bishwaraj.paul@bahash.in 
- Dr. Sahinur Rahman Laskar  
**Role:** Mentor  
Assistant Professor  
School of Computer Science, UPES, Dehradun, India  
**Email:** sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in""")
    footer = """<style>
    .footer-text{
    -webkit-text-size-adjust: 100%;
    -webkit-tap-highlight-color: transparent;
    --blue: #007bff;
    --indigo: #6610f2;
    --purple: #6f42c1;
    --pink: #e83e8c;
    --red: #dc3545;
    --orange: #fd7e14;
    --yellow: #ffc107;
    --green: #28a745;
    --teal: #20c997;
    --cyan: #17a2b8;
    --white: #fff;
    --gray: #6c757d;
    --gray-dark: #343a40;
    --primary: #007bff;
    --secondary: #6c757d;
    --success: #28a745;
    --info: #17a2b8;
    --warning: #ffc107;
    --danger: #dc3545;
    --light: #f8f9fa;
    --dark: #343a40;
    --breakpoint-xs: 0;
    --breakpoint-sm: 576px;
    --breakpoint-md: 768px;
    --breakpoint-lg: 992px;
    --breakpoint-xl: 1200px;
    --font-family-sans-serif: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol";
    --font-family-monospace: SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;
    font-size: 16px;
    font-weight: 400;
    line-height: 24px;
    letter-spacing: 1px;
    font-family: 'Raleway', sans-serif;
    color: #666;
    box-sizing: border-box;
    text-align: center!important;
    }
    @media (min-width: 576px) {
        .col-sm-12 {
            -webkit-box-flex: 0;
            -ms-flex: 0 0 100%;
            flex: 0 0 100%;
            max-width: 100%;
        }
    }
    .row {
        display: -webkit-box;
        display: -ms-flexbox;
        display: flex;
        -ms-flex-wrap: wrap;
        flex-wrap: wrap;
        margin-right: -15px;
        margin-left: -15px;
    }
    @media (min-width: 1200px) {
        .container {
            max-width: 1140px;
        }
    }
    @media (min-width: 992px) {
        .container {
            max-width: 960px;
        }
    }
    @media (min-width: 768px) {
        .container {
            max-width: 720px;
        }
    }
    @media (min-width: 576px) {
        .container {
            max-width: 540px;
        }
    }
    .container {
        width: 100%;
        padding-right: 15px;
        padding-left: 15px;
        margin-right: auto;
        margin-left: auto;
    }
    .footer-bottom-area {
        padding: 30px 0;
        display: block;
        box-sizing: border-box;
    }
    .footer-bottom-bg {
        background: #222;
    }
    </style>
    <footer class="footer-bottom-area footer-bottom-bg">
        <div class="container">
            <div class="row">
                <div class="col-sm-12">
                    <div class="footer-text">
                        <p style="color: white; font-style: sans-serif;"><span>Bahash Private Limited</span> ©2024 - All Right Reserved.</p>
                    </div>
                </div>
            </div>
        </div>
    </footer>
    """
    components.html(footer)
    # Handling query parameters
    query = st.query_params
    try:
        ## Look-up the tab from the query
        if "tab" in query:
            index_tab = query["tab"]
            ## Click on that tab
            js = f"""
            <script>
                var tab = window.parent.document.getElementById('{index_tab}');
                tab.click();
            </script>
            """
            st.components.v1.html(js)

    except ValueError:
        ## Do nothing if the query parameter does not correspond to any of the tabs
        pass

if __name__ == '__main__':
    main()
