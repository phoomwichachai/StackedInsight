# StackedInsight

**StackedInsight** เป็นโปรเจกต์ตัวอย่างสำหรับ **Advanced Supervised Learning** โดยใช้ Python และ scikit-learn เพื่อแสดงการทำงานของ Machine Learning ขั้นสูง ตั้งแต่การสร้างข้อมูล, preprocessing, feature engineering, model tuning, ensemble และ evaluation

---

## 🔹 Features

- สร้าง **synthetic dataset** (numerical + categorical features) พร้อม **missing values** และ **class imbalance**
- Preprocessing ขั้นสูง:
  - Imputation สำหรับ missing values
  - One-hot encoding สำหรับ categorical features
  - Scaling และ Polynomial Features
- **Modeling**
  - Base learners: RandomForestClassifier และ GradientBoostingClassifier
  - Hyperparameter tuning ด้วย RandomizedSearchCV
  - **Stacking ensemble** ใช้ LogisticRegression เป็น meta-model
- Evaluation:
  - ROC AUC
  - Precision-Recall AUC
  - Confusion matrix และ classification report
- Feature importance visualization
- รองรับ scikit-learn ≥1.7

---

## 🛠️ Installation

ติดตั้ง dependencies ด้วย pip:

```bash
pip install scikit-learn pandas matplotlib joblib
