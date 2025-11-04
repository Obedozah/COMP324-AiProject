# ML-based Network Intrusion Detection System (IDS): Proof of Concept
<br>

This project implements a machine learning-based IDS using the *NSL-KDD dataset*. The system classifies network flows as normal or malicious, demonstrating a proof-of-concept for detecting attacks like port scans, probes, and other intrusion types.<br>

# Features

**Dataset**: NSL-KDD (train [20%]/test split, ARFF format)

**Preprocessing**: Encoding categorical features, scaling numeric features, handling missing values

**ML Model**: Random Forest Classifier

**Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

**Trustworthy AI elements**: Feature importance inspection, basic fairness checks, robustness testing with optional lab flows

**Inference**: Predict new flows from CSV/ARFF input using the trained model<br>

# Project Structure

`preprocess_data.py #Loading, cleaning, encoding, scaling`\ 
`train_model.py #Train classifiers, evaluate, save trained models`\ 
`run_inference.py #Loads testing models, predicting/outputs results`

# Installing / Environment Setup

**1. Clone the repository:**
```
git clone <your-repo-url>
cd <repo-folder>
```

**2. Create a virtual environment**:
```
python -m venv sklearn-env
```

**3. Activate the virtual environment**:

&nbsp;&nbsp;Linux / Mac:
&nbsp;&nbsp;```
source sklearn-env/bin/activate
&nbsp;&nbsp;```

&nbsp;&nbsp;Windows (PowerShell):
&nbsp;&nbsp;```
sklearn-env\Scripts\Activate.ps1
&nbsp;&nbsp;```

**4. Install required packages**:
```
pip install -r requirements.txt
```

**5. Verify installation (optional)**:
```
pip list
```