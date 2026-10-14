# 🛡️ Sentinel AI — Spam Message Detection

**An NLP-powered application that detects spam messages using Machine Learning.**

Sentinel AI is a text classification project that uses Natural Language Processing (NLP) and machine learning to classify SMS messages as **Spam** or **Ham (Legitimate)**. It uses TF-IDF for text feature extraction and Logistic Regression for classification, with an interactive web interface built using Streamlit.

## 🚀 Features

- **Spam Detection:** Classifies messages as spam or legitimate.
- **NLP Text Processing:** Converts text messages into numerical features using TF-IDF.
- **Machine Learning:** Uses a trained Logistic Regression classifier.
- **Prediction Confidence:** Displays model confidence scores where available.
- **Interactive Web Interface:** Analyze messages through a user-friendly Streamlit dashboard.
- **Quick Test Messages:** Test the application using sample messages.
- **Real-Time Analysis:** Enter a message and get a prediction without manually running the prediction script.

## 🧠 Technologies Used

- **Python** — Core programming language
- **Natural Language Processing (NLP)** — Text classification
- **Pandas & NumPy** — Data handling
- **Scikit-learn** — TF-IDF vectorization and Logistic Regression
- **Joblib** — Saving and loading the trained model
- **Streamlit** — Interactive web application
- **Git & GitHub** — Version control and project hosting

## 📂 Project Structure

```text
Spam-Mail-Detection/
├── app.py                     # Streamlit web application
├── train.py                   # Model training script
├── predict.py                 # Interactive message prediction
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── models/
│   └── spam_model.joblib      # Saved trained model/pipeline
└── data/
    └── SMSSpamCollection      # SMS spam dataset, if stored locally
```

*Note: The exact structure may vary depending on where you store the dataset and model files.*

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**, which contains SMS messages labeled as spam or ham.

- **Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Task:** Binary text classification
- **Labels:**
  - `ham` — Legitimate message
  - `spam` — Spam message

## ⚙️ How It Works

1. **Data Loading:** Load the labeled SMS dataset.
2. **Data Preparation:** Clean and prepare the message text and labels.
3. **Feature Extraction:** Use TF-IDF to transform text into numerical features.
4. **Model Training:** Train a Logistic Regression classifier on the extracted features.
5. **Model Saving:** Save the trained model or pipeline using Joblib.
6. **Prediction:** Transform a new message using the trained pipeline and predict its class.
7. **Visualization:** Display the prediction and available confidence scores in the Streamlit interface.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sentinel-ai-spam-detector.git
cd sentinel-ai-spam-detector
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application should open in your browser. If it does not, open the local URL displayed in the terminal.

## 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**.

1. Push the project to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Select your repository and the `main` branch.
5. Set the main file path to `app.py`.
6. Click **Deploy**.

Once deployment finishes, Streamlit will provide a public URL for the application.

## 🧪 Example Predictions

| Message | Expected classification |
|---|---|
| "Congratulations! You have won a free prize. Claim now!" | Spam |
| "URGENT! Claim your reward by clicking the link." | Spam |
| "Hey, are we meeting at college today?" | Ham |
| "I'll call you when I reach home." | Ham |

*These are illustrative examples; actual predictions depend on the trained model.*

## 🎯 Learning Outcomes

Through this project, I explored:

- Fundamental NLP concepts and text classification
- TF-IDF and numerical representation of text
- Supervised machine learning with Logistic Regression
- Training, saving, and loading a model pipeline
- Building an interactive ML application with Streamlit
- Version control and deployment using GitHub

## 🔮 Future Improvements

- Compare Logistic Regression with Multinomial Naive Bayes and Linear SVM.
- Add model evaluation metrics such as precision, recall, F1-score, and a confusion matrix.
- Expand the dataset and improve general