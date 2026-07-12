
---
license: mit
library_name: scikit-learn
tags:
- tabular-classification
- tourism
- wellness-tourism
- customer-targeting
- mlops
---

# Wellness Tourism Package Purchase Prediction Model

## Business Objective

This model predicts whether a customer is likely to purchase the new Wellness Tourism Package before the sales team contacts the customer.

## Target Variable

`ProdTaken`

- 0: Customer did not purchase the package
- 1: Customer purchased the package

## Best Model

Random Forest

## Recommended Business Threshold

0.4

## Metrics at Recommended Threshold

- Accuracy: 0.8667
- Precision: 0.6224
- Recall: 0.7871
- F1 score: 0.6952
- ROC-AUC: 0.9306
- Average precision: 0.7877

## Intended Use

The model should be used as a campaign targeting and lead-prioritization tool for the marketing and sales teams.

## Artifacts

- `wellness_tourism_model.joblib`
- `feature_schema.json`
- `model_metadata.json`
- `experiment_results.csv`
- `threshold_analysis.csv`
- `classification_report.txt`
