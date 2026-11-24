# Portfolio Project Data — Diabetes Classification Using KNN with Data Normalization

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/diabetes_classification_using_knn/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image04.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "diabetes-classification-using-knn-with-data-normalization",
  "code": "PRO-44 // ML-26",
  "title": "Diabetes Classification Using KNN with Data Normalization",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2025",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Diabetes-Classification-Using-KNN-with-Data-Normalization",
  "license": "MIT License",
  "banner": "/images/projects/diabetes_classification_using_knn/banner.webp",
  "summary": "Empirical KNN diabetes classification benchmark comparing Min-Max, Z-Score, and Decimal Scaling across k values with paired t-test validation.",
  "projectDescription": "Supervised tabular benchmark evaluating how feature scaling transforms K-Nearest Neighbors classification on Pima Indians Diabetes data. Combines Random Forest feature selection with four normalization regimes across k=1..11, achieving 81.17% test accuracy and verified statistical significance.",
  "architectureSummary": "CSV Ingestion -> 80/20 Split -> Random Forest Top-4 Selection -> 4 Scaling Pipelines -> KNN k-Sweep (1..11) -> Paired t-Test",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Jupyter Notebook"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised Classification Benchmark",
    "targetPlatform": "Python / Jupyter Notebook",
    "primaryDomain": "Clinical Tabular Diagnostics",
    "architecturePipeline": "RF Feature Selection + Scaled KNN",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2025"
  },
  "mlProjectInfo": {
    "projectType": "Predictive Classification Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Tabular Clinical Risk",
    "projectStatus": "Completed",
    "developmentYear": "2025",
    "capabilityBadges": [
      "Distance-Based Classification",
      "Feature Scaling Benchmark",
      "Paired Statistical Audit"
    ]
  },
  "mlProblemContext": "Distance-based classifiers like K-Nearest Neighbors depend heavily on feature magnitudes, causing high-range clinical variables to dominate Euclidean distance over smaller-scale predictors. This study systematically isolates how normalization reshapes neighborhood boundaries and improves diabetes prediction reliability.",
  "mlDatasetOverview": {
    "profileBadge": "Pima Clinical Tabular",
    "source": "Local Pima Indians Diabetes CSV (diabetes.csv)",
    "modality": "Tabular",
    "scale": "768 Records (614 Train / 154 Test)",
    "inputShape": "N x 8 Raw Features -> N x 4 Selected",
    "dataFormat": "CSV",
    "description": "Contains 768 patient diagnostic records with zero nulls and zero duplicates, partitioned via an 80/20 split (random_state=42) into 614 training and 154 test samples (99 non-diabetic, 55 diabetic).",
    "provenanceNote": "MIT-licensed benchmark workflow operating on the public Pima Indians Diabetes tabular schema stored locally as diabetes.csv."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Clinical Feature & Target Space",
    "spaceNote": "Top 4 features selected from 8 numerical clinical attributes via Random Forest Gini importance on the training split.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Outcome",
      "type": "Binary Class",
      "distribution": "500 Negative (65.10%) / 268 Positive (34.90%)"
    },
    "inputSignals": [
      {
        "name": "Glucose",
        "type": "Numerical",
        "importanceOrRole": "Rank 1 (Importance: 0.258864)",
        "description": "Plasma glucose concentration measuring glycemic regulation; strongest individual discriminator for diabetes risk."
      },
      {
        "name": "BMI",
        "type": "Numerical",
        "importanceOrRole": "Rank 2 (Importance: 0.169984)",
        "description": "Body mass index reflecting anthropometric metabolic risk and insulin resistance correlation."
      },
      {
        "name": "Age",
        "type": "Numerical",
        "importanceOrRole": "Rank 3 (Importance: 0.140931)",
        "description": "Patient age in years capturing cumulative metabolic susceptibility."
      },
      {
        "name": "DiabetesPedigreeFunction",
        "type": "Numerical",
        "importanceOrRole": "Rank 4 (Importance: 0.123768)",
        "description": "Hereditary diabetes risk score synthesizing familial medical history."
      }
    ],
    "dataCharacteristicsSummary": "Clinical predictors span widely incompatible scales, from DiabetesPedigreeFunction decimals to three-digit Glucose and Insulin values. Training-only Random Forest ranking concentrates 69.35% of total Gini importance within the top four retained features."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Binary Classification via Instance-Based Learning",
    "methodologyDescription": "Systematically evaluates K-Nearest Neighbors across six odd neighbor counts (k = 1, 3, 5, 7, 9, 11) under four parallel scaling conditions: unscaled baseline, Min-Max Scaling, Z-Score Standardization, and Decimal Scaling to isolate feature scaling effects.",
    "justification": "KNN makes no parametric distribution assumptions and directly reveals how feature rescaling alters Euclidean distance geometry, enabling rigorous paired t-test comparison against an unscaled control."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Train-Test Partitioning",
      "description": "Split the 768-row dataset 80/20 using random_state=42 before any feature selection or scaling to isolate 614 training and 154 test samples.",
      "badge": "80/20 Split"
    },
    {
      "step": "02",
      "title": "Training-Only Feature Selection",
      "description": "Fit RandomForestClassifier(random_state=42) on training features and drop BloodPressure, Pregnancies, Insulin, and SkinThickness, retaining the top 4 predictors.",
      "badge": "Top-4 RF"
    },
    {
      "step": "03",
      "title": "Parallel Feature Normalization",
      "description": "Transform the 4 selected features independently using MinMaxScaler, StandardScaler (fit on train, applied to test), and custom base-10 Decimal Scaling.",
      "badge": "4 Scaling Arms"
    }
  ],
  "mlDataPreparationNote": "Train-test split and Random Forest feature selection occur strictly prior to MinMaxScaler and StandardScaler fitting on training data; note that Decimal Scaling computes column exponents per split.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "K-Nearest Neighbors (KNeighborsClassifier)",
    "architectureSummary": "Non-parametric instance-based classifier assigning binary labels via majority vote over the k closest training vectors in 4-dimensional scaled feature space using Euclidean (Minkowski p=2) distance.",
    "configurationBadge": "KNN k-Sweep [1..11]",
    "hyperparameters": [
      {
        "name": "n_neighbors",
        "value": "[1, 3, 5, 7, 9, 11]",
        "description": "Swept across six odd values to eliminate tie votes and trace bias-variance trade-offs."
      },
      {
        "name": "metric / weights",
        "value": "Minkowski (p=2) / uniform",
        "description": "Scikit-Learn default Euclidean distance with equal voting weight across selected neighbors."
      },
      {
        "name": "rf_random_state",
        "value": "42",
        "description": "Fixed seed for train_test_split and RandomForestClassifier feature importance ranking."
      }
    ],
    "objectiveCriterion": "Non-parametric neighborhood majority vote (Euclidean proximity)",
    "optimizationSolver": "Lazy instance indexing via Scikit-Learn KNeighborsClassifier.fit",
    "trainingProcess": "Each experiment trains six KNN estimators across odd k values from 1 to 11 on the 614-sample training matrix. Because KNN stores training vectors rather than running gradient epochs, training completes deterministically and instantaneously across all 24 experimental models."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Holdout Test Evaluation & Paired t-Test",
    "description": "All 24 models are evaluated on the held-out 154-sample test split using Accuracy, macro-averaged Precision, Recall, F1-Score, Specificity, ROC-AUC, and Confusion Matrices, followed by paired t-tests (alpha = 0.05) across k values.",
    "metrics": [
      {
        "label": "BEST TEST ACCURACY",
        "value": "81.17%",
        "benchmark": "Baseline: 75.97%"
      },
      {
        "label": "MACRO F1-SCORE (K=7)",
        "value": "79.58%",
        "benchmark": "Baseline: 74.14%"
      },
      {
        "label": "SPECIFICITY (K=7)",
        "value": "84.85%",
        "benchmark": "Baseline: 79.80%"
      },
      {
        "label": "PEAK ROC-AUC (Z-SCORE)",
        "value": "0.8129",
        "benchmark": "Baseline: 0.7802"
      },
      {
        "label": "MIN-MAX ACC P-VALUE",
        "value": "0.01035",
        "benchmark": "Threshold < 0.05"
      },
      {
        "label": "MIN-MAX AUC P-VALUE",
        "value": "0.00003",
        "benchmark": "Threshold < 0.05"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Comparison of Evaluation Metrics Across Normalization Techniques and K Values",
      "badge": "Multi-Metric Grid",
      "imageUrl": "/images/projects/diabetes_classification_using_knn/image01.webp",
      "telemetryNote": "Min-Max Scaling peaks at k=7 (81.17% accuracy, 79.58% macro F1); Z-Score reaches 0.8129 ROC-AUC at k=11."
    },
    {
      "title": "Feature Importance Using Random Forest (Train Data)",
      "badge": "Feature Ranking",
      "imageUrl": "/images/projects/diabetes_classification_using_knn/image02.webp",
      "telemetryNote": "Glucose (0.258864), BMI (0.169984), Age (0.140931), and DiabetesPedigreeFunction (0.123768) lead all 8 features."
    },
    {
      "title": "Paired T-Test P-Value Across Evaluation Metrics",
      "badge": "Hypothesis Test",
      "imageUrl": "/images/projects/diabetes_classification_using_knn/image03.webp",
      "telemetryNote": "Min-Max Scaling achieves p < 0.05 across all 6 metrics against the unscaled baseline."
    },
    {
      "title": "Paired T-Statistic Across Evaluation Metrics",
      "badge": "Effect Direction",
      "imageUrl": "/images/projects/diabetes_classification_using_knn/image04.webp",
      "telemetryNote": "Negative t-statistics (-3.54 to -14.31) confirm consistent Min-Max gains; positive Decimal values reflect degradation."
    }
  ],
  "mlVisualizationTelemetryNote": "Evaluation metrics are measured exclusively on the held-out 154-row test set (no training-set accuracy conflation); paired t-tests compare all 6 k-values per metric.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Equalizing feature ranges via Min-Max Scaling prevents high-magnitude predictors like Glucose from drowning out BMI, Age, and DiabetesPedigreeFunction in Euclidean distances, enabling balanced k=7 neighborhood boundaries with statistically significant gains.",
    "keyDrivers": [
      {
        "factor": "Min-Max Feature Rescaling",
        "impact": "Lifted peak test accuracy from 75.97% (raw k=11) to 81.17% (Min-Max k=7) with p < 0.05 across all six metrics."
      },
      {
        "factor": "Glucose & BMI Dominance",
        "impact": "Combined Random Forest importance of 0.428848 anchored the 4-feature neighborhood representation."
      },
      {
        "factor": "Optimal Neighborhood Size (k = 7)",
        "impact": "Balanced local sensitivity and noise smoothing, yielding TN=84, FP=15, FN=14, TP=41 under Min-Max Scaling."
      }
    ],
    "limitations": [
      {
        "title": "No Serialized Model Checkpoints",
        "description": "Trained KNN instances remain in notebook memory and are not exported as .pkl or .joblib artifacts."
      },
      {
        "title": "Unimputed Clinical Zero Values",
        "description": "Physiologically implausible zeros present in the raw dataset (e.g., Insulin, SkinThickness, BloodPressure) were not imputed prior to selection."
      },
      {
        "title": "Decimal Scaling Exponent & Recall Entry",
        "description": "Decimal Scaling calculates column exponents per split rather than reusing train exponents, and the t-test notebook lists Min-Max k=1 recall as 0.7121 vs. 0.7212."
      }
    ],
    "futureImprovements": [
      {
        "title": "Stratified K-Fold Cross-Validation",
        "target": "Replace single 80/20 split with 5-fold or 10-fold CV for variance estimation."
      },
      {
        "title": "Domain-Aware Missing Zero Imputation",
        "target": "Impute biologically invalid zero entries using training-only medians inside a Scikit-Learn Pipeline."
      },
      {
        "title": "Model Serialization & CLI Inference",
        "target": "Export the winning MinMaxScaler + KNN (k=7) pipeline with joblib for standalone prediction."
      }
    ],
    "discoveryNote": "All feature importance rankings and significance claims are grounded directly in notebook execution logs."
  },
  "mlModelArtifacts": [
    {
      "name": "Min-Max Scaling/diabetes_train_scaled.csv",
      "format": "CSV"
    },
    {
      "name": "Min-Max Scaling/diabetes_test_scaled.csv",
      "format": "CSV"
    },
    {
      "name": "Z-Score (Standard Scaling)/diabetes_train_scaled.csv",
      "format": "CSV"
    },
    {
      "name": "Decimal Scaling/diabetes_train_scaled.csv",
      "format": "CSV"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Learning",
    "modelArchitecture": "K-Nearest Neighbors",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Holdout & Paired t-Test"
  }
}
```
