from pathlib import Path
import joblib

# Locate the trained model
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_model.joblib"

# Load the model and assign it to spam_model
spam_model = joblib.load(MODEL_PATH)

print("=== NLP Spam Mail Detector ===")
print("Type 'exit' to stop.")
while True:
    message=input("\nEnter a message: ").strip()

    if message.lower()=="exit":
        break
    if not message:
        print("Please Enter a message.")
        continue

    prediction=int(spam_model.predict([message])[0])
    probabilities=spam_model.predict_proba([message])[0]

    if prediction==1:
        print("Result: SPAM")
    else:
        print("Result: LEGITIMATE")

    print(f"Model Confidence: {max(probabilities)*100:.1f}%")