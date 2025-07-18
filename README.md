# 🎓 Student Admission Predictor - Logistic Regression (Streamlit App)

![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

Predict your chances of graduate school admission using a simple machine learning model!  
This project implements logistic regression from scratch (without using libraries like `sklearn.linear_model`) and wraps it in a clean and interactive **Streamlit web app**.

## 🔗 Live Demo

▶️ [Try the App](https://student-admission-predictor-minimal-version.streamlit.app/)

---

## 📌 Features

- Logistic Regression implemented **from scratch in NumPy**
- User-friendly **Streamlit interface** for predictions
- Predicts **chances of admission (0 or 1)** based on:
  - GRE Score
  - TOEFL Score
  - University Rating
  - SOP Strength
  - LOR Strength
  - CGPA
  - Research Experience

---

## 📊 Dataset

- **Source:** [Admission Prediction Dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)
- **Total Samples:** 500 students
- **Label Used:** `Chance of Admit ≥ 0.75 → Admitted (1)`, otherwise `Not Admitted (0)`

---

## 🧠 Model Highlights

- Custom **sigmoid**, **loss**, **gradient**, and **training loop**
- Uses **min-max normalization**
- Evaluation metrics:
  - Accuracy: 87.20%
  - Precision: 84.68%
  - Recall: 86.24%
  - F1 Score: 85.45%

---

## 🚀 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/shivamr021/student-admission-predictor.git
cd student-admission-predictor
````

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## 📁 File Structure

```bash
.
├── streamlit_app.py          # Streamlit app (interactive UI)
├── admission_data.csv        # Dataset (used for training)
├── README.md                 # This file
├── requirements.txt          # Dependencies
```

---

## 💡 Learning Outcome

* Implemented end-to-end **binary classification** without using machine learning libraries
* Learned to build a basic **ML app with Streamlit**
* Understood core ML evaluation metrics
* Practiced data preprocessing and normalization

---

## 🌐 Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Streamlit

---

## ✨ Future Improvements

* Add model trained using `scikit-learn` for comparison
* Allow file upload to test batch data
* Deploy advanced model on platforms like HuggingFace or Render

---

## 👤 Author

**Shivam Rathod**
📬 [LinkedIn](https://linkedin.com/in/shivamrathod021)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

