# SMS Spam Detection using Custom NLP Library

## Overview
This project involves detecting spam messages using a custom NLP library, integrated with NLTK. The model processes the SMS Spam and Harm dataset, applying advanced text preprocessing techniques and Random Forest Classifier for classification. The final model achieves a 97% accuracy on the test data.

## Features
- **Text Preprocessing Pipeline**:
  - Cleans the input text by removing unwanted patterns such as URLs, numbers, and special characters.
  - Tokenizes the input into sentences.
  - Removes stop words.
  - Applies lemmatization with accurate Part-of-Speech (POS) tagging for better performance.
  
- **Vectorization**:
  - Uses Word2Vec for word embedding, converting words into numerical vectors.
  - Implements Average Word2Vec for better computational efficiency.

- **Classification**:
  - Uses a Random Forest Classifier to predict whether a message is spam or not.
  - Achieves 97% accuracy on the test dataset.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


    ```

## Project Structure

- **custom_library**: Contains the custom text preprocessing library integrated with NLTK.
- **data**: Includes the SMS spam and harm dataset.
- **notebooks**: Jupyter notebooks with model training, evaluation, and analysis.


## Results
- **Accuracy**: 97% on the test dataset.
- **Preprocessing Efficiency**: Improved through Part-of-Speech tagging and lemmatization, ensuring accurate text representation before classification.

## Requirements
- Python 3.10
- NLTK
- Gensim (for Word2Vec)
- Scikit-learn
- Pandas
- NumPy

## How to Run
1. Ensure all dependencies are installed using the `requirements.txt` file.
2. Use the `notebooks` to run preprocessing and classification tasks.



