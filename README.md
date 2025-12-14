# 📨 SMS Spam Message Detection using Machine Learning

This project implements a machine learning model to classify SMS messages as **Spam** or **Ham (Not Spam)** using **Linear Regression**. The model is trained and evaluated on the **SMS Spam Collection Dataset**.

---

## 🔍 Project Overview

Spam message detection is a classic **text classification** problem in machine learning.
In this project, SMS messages are analyzed, processed, and classified based on their textual content.

### Key Highlights:

* Text preprocessing and cleaning
* Feature extraction using vectorization techniques
* Spam classification using **Logistic Regression**
* Performance evaluation on test data

---

## 📁 Dataset

**SMS Spam Collection Dataset**

* Total messages: **5,574**
* Labels:

  * `spam` – unwanted promotional or fraudulent messages
  * `ham` – legitimate messages

Each record contains:

* Message label
* Raw SMS text

🔗 Dataset link:
[https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

---

## 🧠 Methodology

### 1️⃣ Data Loading

The dataset is loaded using Pandas and inspected for structure and class distribution.

### 2️⃣ Data Preprocessing

* Convert text to lowercase
* Remove punctuation and unnecessary symbols
* Encode labels (`spam = 1`, `ham = 0`)

### 3️⃣ Feature Extraction

Text messages are converted into numerical form using **vectorization** techniques such as:

* Bag of Words / CountVectorizer
* TF-IDF

### 4️⃣ Model Training

* A **Logistic Regression** model is trained on the vectorized text

### 5️⃣ Model Evaluation

The model is evaluated using:

* Accuracy
* Confusion Matrix

---

## 📊 Sample Prediction

```python
message = "Congratulations! You've won a free prize"
prediction = model.predict(vectorizer.transform([message]))

if prediction >= 0.5:
    print("Spam")
else:
    print("Ham")
```

---

## 📈 Results

* Achieved good classification accuracy on test data
* Demonstrates the effectiveness of basic ML models with proper text preprocessing

---

## 📂 Project Structure

```
Spam-Message-Detection-using-ML/
│
├── spam_message_detection.ipynb
├── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/M-S-Abhiram-Verma/Spam-Message-Detection-using-ML.git
```

2. Navigate to the project directory:

```bash
cd Spam-Message-Detection-using-ML
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open the Jupyter Notebook and run all cells.

---

## 👤 Author

**Sathya Abhiram Verma**
GitHub: [https://github.com/M-S-Abhiram-Verma](https://github.com/M-S-Abhiram-Verma)

---

## 📚 References

* SMS Spam Collection Dataset – UCI / Kaggle
  [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
