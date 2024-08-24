# Indic Multilingual Text Summarization

## Introduction

Indic Multilingual Text Summarization is a machine learning project aimed at summarizing in text across multiple Indic languages, starting with Bengali. This repository hosts the Streamlit code for easy deployment and usage. Download model files at [https://uploadnow.io/f/dc4f6Hb](https://uploadnow.io/f/dc4f6Hb)

## Project Structure

- `models/`: Directory containing trained models for emotion detection. Download model files at [https://uploadnow.io/f/dc4f6Hb](https://uploadnow.io/f/dc4f6Hb)
- `app.py`: Streamlit application code for interacting with the models.
- `requirements.txt`: List of Python dependencies required to run the Streamlit app.

## Tokenizers

The dataset text is converted to tokens using the keras tokenizer.

### Detailed Model Descriptions

#### LSTM
- **Description**: 
- **Performance**: 

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
4. **Download model files at [https://uploadnow.io/f/dc4f6Hb](https://uploadnow.io/f/dc4f6Hb) and place them in a folder named models inside the project directory.**
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
