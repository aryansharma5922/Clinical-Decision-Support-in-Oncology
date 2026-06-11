# B.Tech Final Year Project Report

## Title
Clinical Decision Support in Oncology: Early Detection of Malignant Tumours from Diagnostic Imaging Features Using PCA, Random Forest, and SVM Voting Ensemble

## Abstract
This project develops a machine learning based clinical decision support system for breast tumor classification. The system uses the real Breast Cancer Wisconsin Diagnostic dataset from the UCI Machine Learning Repository, containing 569 samples and 30 diagnostic imaging features. The workflow includes data cleaning, exploratory data analysis, feature scaling, PCA dimensionality reduction, Random Forest, SVM, soft voting ensemble learning, model comparison, SHAP explainability, and Streamlit deployment.

## Literature Review
Breast cancer diagnosis has been widely studied using classical machine learning methods and medical imaging descriptors. Prior research shows that morphology-related features such as radius, perimeter, area, compactness, concavity, and texture can discriminate between malignant and benign tumors. Random Forest models are robust for tabular biomedical features due to their ensemble structure and feature ranking capability. Support Vector Machines are effective in high-dimensional spaces and can model nonlinear decision boundaries through RBF kernels. PCA reduces multicollinearity and retains major variance patterns. Explainable AI methods such as SHAP improve trust by presenting contribution patterns.

## Methodology
The WDBC dataset is loaded and cleaned. Diagnosis labels are encoded as Malignant = 1 and Benign = 0. Data quality checks include null value analysis and duplicate removal. Exploratory analysis includes class distribution, correlation heatmap, histograms, pair plot, box plots, and feature importance visualization. StandardScaler normalizes all thirty input features. PCA retains 95 percent cumulative variance. Random Forest and SVM classifiers are trained on PCA-transformed features, followed by a soft voting ensemble that combines predicted probabilities. Metrics are computed on a stratified test set, and the best model is selected automatically.

## Algorithms Used
### Principal Component Analysis
PCA transforms correlated diagnostic features into orthogonal principal components while retaining 95 percent of explained variance. This reduces dimensionality and helps downstream classifiers handle multicollinearity.

### Random Forest
Random Forest is an ensemble of decision trees trained using bootstrap sampling. It improves generalization by averaging multiple trees and provides feature importance estimates.

### Support Vector Machine
The SVM uses an RBF kernel to learn nonlinear boundaries between malignant and benign classes. Probability estimates are enabled for ensemble voting and clinical confidence scoring.

### Soft Voting Ensemble
The soft voting ensemble averages class probabilities from Random Forest and SVM. The combined model benefits from complementary decision boundaries and improves robustness.

### SHAP Explainability
SHAP is used to explain global and individual prediction behavior. Summary and waterfall plots help interpret model decisions for principal components and derived feature contribution patterns.

## Results and Discussion
Model performance is evaluated using accuracy, precision, recall, F1 score, ROC AUC, sensitivity, and specificity. Confusion matrix, ROC curve, and precision-recall curve artifacts are generated. The dashboard displays comparison tables and charts for Random Forest, SVM, and Voting Ensemble. The final prediction interface provides a class label, confidence score, malignant probability, and low/medium/high risk category.

## Conclusion
The project demonstrates a complete AI-enabled oncology decision support system using a real medical dataset. It combines reliable preprocessing, dimensionality reduction, ensemble learning, explainability, and an interactive Streamlit interface. The system is suitable as an academic final year project and as a prototype for further clinical validation.

## References
1. UCI Machine Learning Repository, Breast Cancer Wisconsin Diagnostic Dataset, https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. W. H. Wolberg, W. N. Street, and O. L. Mangasarian, Breast Cancer Wisconsin Diagnostic Dataset.
3. Breiman, L. Random Forests. Machine Learning, 2001.
4. Cortes, C. and Vapnik, V. Support-Vector Networks. Machine Learning, 1995.
5. Jolliffe, I. T. Principal Component Analysis. Springer.
6. Lundberg, S. M. and Lee, S.-I. A Unified Approach to Interpreting Model Predictions. NeurIPS, 2017.
