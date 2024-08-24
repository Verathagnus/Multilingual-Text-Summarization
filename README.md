# Indic Multilingual Text Summarization

## Introduction

Indic Multilingual Text Summarization is a machine learning project aimed at summarizing in text across multiple Indic languages, starting with Bengali. This repository hosts the Streamlit code for easy deployment and usage. Download model files at [https://uploadnow.io/f/8YZ7pBb](https://uploadnow.io/f/8YZ7pBb)

## Project Structure

- `models/`: Directory containing trained models for emotion detection. Download model files at [https://uploadnow.io/f/8YZ7pBb](https://uploadnow.io/f/8YZ7pBb)
- `app.py`: Streamlit application code for interacting with the models.
- `requirements.txt`: List of Python dependencies required to run the Streamlit app.

## Tokenizers

The dataset text is converted to tokens using the keras tokenizer.

### Detailed Model Descriptions

#### LSTM
- **Description**: The Long Short-Term Memory (LSTM) model is a type of recurrent neural network (RNN) that is well-suited for processing sequential data, such as text. In the context of this project, the LSTM model is used to process and understand the sequence of words or tokens in the input text, enabling it to generate a meaningful summary. LSTMs are particularly effective at capturing long-term dependencies in text, which is essential for summarizing content that spans multiple sentences or even paragraphs. This model has been fine-tuned on a multilingual corpus, starting with Bengali, to handle the nuances of different Indic languages.
- **Performance**: The LSTM model demonstrates robust performance in generating summaries in Bengali. While it excels at maintaining the context of the original text, its performance can vary depending on the complexity and length of the input. <!--The model has been evaluated on a variety of test cases, showing an average ROUGE score of [insert score here], which indicates a good balance between precision and recall in the generated summaries. The model performs well with moderate-length texts but may require further fine-tuning for extremely short or long documents.-->

## Setup

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/<yourusername>/multilingual-text-summarization.git
   cd multilingual-text-summarization
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Download model files at [https://uploadnow.io/f/8YZ7pBb](https://uploadnow.io/f/8YZ7pBb) and place them in a folder named models inside the project directory.**
5. **Run the Streamlit application:**
   ```sh
   streamlit run streamlit_app.py
   ```

## Usage

Once the Streamlit app is running, you can interact with the models through a web interface. You can input text in Bengali (and other supported Indic languages in the future) and select the model you wish to use for text summarization. The app will display the summarized text. Using fasttext-langdetect and iso639 packages, the input language is found and then appropriate models list are selected for that language.  

<!-- ## Example

![Streamlit App Screenshot](screenshot.png)  # Add a screenshot of your Streamlit app here -->

## Future Work

- **Extend Language Support**: Increase the number of Indic languages supported by training the models on additional language datasets.
- **Improve Model Performance**: Experiment with advanced models and techniques to enhance the accuracy and robustness of emotion detection.
- **User Feedback**: Incorporate user feedback to continuously improve the application and its usability.

## Contributors

- <h2>Bishwaraj Paul</h2>
  <p><strong>Role: </strong>Intern<br>
  Email: bishwaraj.paul98@gmail.com / bishwaraj.paul@bahash.in<br>
  </p>
- <h2>Dr. Sahinur Rahman Laskar</h2>
  <p><strong>Role:</strong> Mentor<br>
  Assistant Professor<br>
  School of Computer Science, UPES, Dehradun, India<br>
  Email: sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in<br>
  </p>
---
