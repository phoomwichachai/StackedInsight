# advanced_supervised_fast.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
X_num, y = make_classification(n_samples=2000, n_features=10, n_informative=6, n_redundant=2,
                               n_clusters_per_class=2, weights=[0.85, 0.15], flip_y=0.01, random_state=42)

num_cols = [f"num_{i}" for i in range(X_num.shape[1])]
df = pd.DataFrame(X_num, columns=num_cols)
df['target'] = y
n = df.shape[0]
df['cat_1'] = rng.choice(['A','B','C'], size=n, p=[0.6,0.3,0.1])
df['cat_2'] = rng.choice(['X','Y'], size=n, p=[0.7,0.3])
df['ord_1'] = rng.integers(0, 5, size=n)

for col in num_cols + ['cat_1']:
    mask = rng.random(n) < 0.05
    df.loc[mask, col] = np.nan

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

numeric_features = num_cols + ['ord_1']
categorical_features = ['cat_1', 'cat_2']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ('scaler', StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features),
], remainder='drop', verbose_feature_names_out=False)

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(random_state=42)

rf_param_dist = {'n_estimators': [100,200], 'max_depth': [5,12,None], 'min_samples_split': [2,5], 'min_samples_leaf':[1,2]}
gb_param_dist = {'n_estimators': [100,200], 'learning_rate':[0.01,0.05,0.1], 'max_depth':[3,5], 'subsample':[0.8,1.0]}

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

rf_search = RandomizedSearchCV(rf, rf_param_dist, n_iter=4, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1)
gb_search = RandomizedSearchCV(gb, gb_param_dist, n_iter=4, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1)

pipe_rf = Pipeline([('pre', preprocessor), ('clf', rf_search)])
pipe_gb = Pipeline([('pre', preprocessor), ('clf', gb_search)])

pipe_rf.fit(X_train, y_train)
pipe_gb.fit(X_train, y_train)

best_rf = pipe_rf.named_steps['clf'].best_estimator_
best_gb = pipe_gb.named_steps['clf'].best_estimator_

estimators = [('rf', best_rf), ('gb', best_gb)]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1)

stack_pipeline = Pipeline([('pre', preprocessor), ('selector', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42))), ('stack', stack)])
stack_pipeline.fit(X_train, y_train)

y_proba = stack_pipeline.predict_proba(X_test)[:,1]
y_pred = stack_pipeline.predict(X_test)

roc_auc = roc_auc_score(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

print(f"Test ROC AUC: {roc_auc:.4f}")
print(f"Test PR AUC: {pr_auc:.4f}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve (Fast Stacking)")
plt.show()

PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Precision-Recall Curve (Fast Stacking)")
plt.show()

# save
joblib.dump(stack_pipeline, 'fast_stacking_pipeline.joblib')
print("Saved fast_stacking_pipeline.joblib")
