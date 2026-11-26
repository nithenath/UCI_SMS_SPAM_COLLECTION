# SMS Spam Classification Using SimpleRNN

## Project Overview

This project implements an SMS Spam Classification system using Deep Learning and Natural Language Processing (NLP).

A Simple Recurrent Neural Network (SimpleRNN) is trained to classify SMS messages into two categories:

- **Spam:** Unwanted or promotional messages.
- **Ham:** Legitimate messages.

The trained model is integrated with a Streamlit web application, allowing users to enter messages and receive real-time predictions.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow / Keras
- Scikit-learn
- Natural Language Processing (NLP)
- Streamlit

## Project Structure

```text
RNN_implementation/
│
├── Datasets/
│   ├── data/
│   ├── dataset.csv
│   └── dataset.ipynb
│
├── logs/
│
├── sms_spam_classification/
│
├── app.py
├── preprocessing.ipynb
├── spam_model.h5
├── tokenizer.pkl
├── maxlen.pkl
│
├── logging.py
├── exception.py
├── setup.py
├── requirements.txt
└── .gitignore
```

## Workflow

### 1. Data Preprocessing

- Loaded the SMS dataset using Pandas.
- Removed duplicate records.
- Converted messages into lowercase.
- Removed punctuation.
- Prepared text data for tokenization.

### 2. Tokenization and Sequence Preparation

- Used the Keras Tokenizer to convert text into numerical sequences.
- Converted words into integer indices.
- Applied padding using `pad_sequences()` to ensure uniform sequence lengths.

### 3. Model Building

Implemented a SimpleRNN-based neural network using TensorFlow/Keras.

Model architecture:

```text
Input Sequence
      |
      v
Embedding Layer
      |
      v
SimpleRNN (64 Units)
      |
      v
Dense (1 Unit, Sigmoid)
      |
      v
Spam / Ham
```

**Model configuration:**

- Embedding Dimension: 100
- SimpleRNN Units: 64
- Output Activation: Sigmoid
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 10
- Batch Size: 32

### 4. Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Used Scikit-learn's `classification_report` and `confusion_matrix` for evaluation.

### 5. Model Serialization

Saved the trained model and preprocessing artifacts for reuse during deployment.

| File | Purpose |
|---|---|
| `spam_model.h5` | Saved trained neural network |
| `tokenizer.pkl` | Saved tokenizer and vocabulary |
| `maxlen.pkl` | Saved maximum sequence length |

This avoids retraining the model whenever the application starts.

### 6. Streamlit Deployment

Developed a Streamlit application that:

1. Accepts SMS messages from users.
2. Performs the same text preprocessing used during training.
3. Converts messages into numerical sequences.
4. Applies padding.
5. Loads the saved model and generates predictions.
6. Displays whether the message is Spam or Ham.

## Installation and Execution

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Navigate to the Project

```bash
cd RNN_implementation
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

## Additional Modules

The project also contains `logging.py` and `exception.py` as utility modules intended for potential future integration.

They are not currently integrated into the model training or Streamlit prediction workflow.

## Future Improvements

- Experiment with LSTM and GRU architectures.
- Improve model performance through hyperparameter tuning.
- Integrate logging and exception handling into the application.
- Explore additional NLP preprocessing techniques.
- Deploy the application on Streamlit Community Cloud.

## Author

**Nithenath Vairavan**

Computer Science and Engineering | AI & Data Science

---

*This project was developed as part of my hands-on learning in Natural Language Processing and Deep Learning.*
