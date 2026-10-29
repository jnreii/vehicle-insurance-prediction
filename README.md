# Vehicle Insurance Cross-Sell Predictor 🚗

A Streamlit app that predicts whether an existing health insurance customer is likely to be interested in vehicle insurance, so marketing teams can focus outreach on the customers most likely to say yes.

**Live app:** https://vehicle-insurance-prediction-6xip7eynikmvuh6x4o86ky.streamlit.app/

![Single customer prediction](<img width="2738" height="1444" alt="Screenshot 2026-10-29 110447" src="https://github.com/user-attachments/assets/10927518-5c8e-4a7c-bac4-07cbc9012f3a" />)
![Scoring a customer list](<img width="2720" height="1450" alt="Screenshot 2026-10-29 110516" src="https://github.com/user-attachments/assets/f58fafa0-30b0-48ab-aefe-c4bc613f4ba0" />)

## Business Problem

The company already has 381,109 health insurance customers and wants to cross-sell vehicle insurance to them. Calling all of them is expensive and time-consuming, and most of the people picking up the phone might not be interested anyway. Only around 12% are.

So the real problem is not reaching customers, it's about working out which ones are worth reaching. This app scores every customer on how likely they are to say yes and purchase vehicle insurance, which lets the marketing team work down a ranked list instead of blanket-calling the whole base of customers. On the test set, that cuts the list by about 60% while still reaching roughly 9 in 10 of the people who would have converted.

## Dataset

- **Source:** Health Insurance Cross Sell Prediction (Kaggle)
- **Size:** 381,109 rows, no missing values or duplicates
- **Target:** `Response`, heavily imbalanced at roughly 12% interested to 88% not

That imbalance shapes everything downstream. A model that simply predicts "not interested" every time scores 88% accuracy while being completely useless, so accuracy was ruled out as a headline metric from the start.

Only `train.csv` was used. Kaggle's `test.csv` has no target column, so there's nothing to evaluate predictions against.

## Modelling Approach

Three models were compared: Decision Tree, Random Forest, and Gradient Boosted Trees.

The data was split 70/30 with `random_state=2026`, stratified so both sets keep the same 88/12 balance. `Region_Code` and `Policy_Sales_Channel` both have high cardinality, so each was cut down to its top 10 categories with everything else grouped as `Other`. Those categories were worked out from the training set alone, so no test set information leaks back into preprocessing.

Class weighting made the biggest difference by far. With no weighting, one Decision Tree config scored a cross-validated F1 of 0.013. But with `class_weight='balanced'` at the same depth, it scored 0.428.

The Gradient Boosted Tree is the clearest illustration of why accuracy was the wrong metric here. It posted the highest accuracy of any model at 0.88, and found almost no interested customers at all: just 3 out of 14,013 in the test set (F1 ≈ 0.00 on the positive class). It got that score by predicting "not interested" for nearly everyone.

This left Decision Tree and Random Forest, which after tuning finished close on F1 for the interested class (0.43 vs 0.45) but pulled in different directions:

- **Decision Tree:** higher recall at 0.92, so it catches more real prospects but flags more people who won't convert
- **Random Forest:** higher precision at 0.31 vs 0.28, so fewer false alarms but more missed buyers

**Results on the test set** (114,333 customers, 14,013 interested; metrics are for the interested class):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Gradient Boosted Trees (baseline) | 0.88 | 0.38 | 0.00 | 0.00 |
| Random Forest (baseline) | 0.86 | 0.34 | 0.15 | 0.21 |
| Decision Tree (baseline) | 0.82 | 0.29 | 0.30 | 0.30 |
| Random Forest (tuned) | 0.75 | 0.31 | 0.81 | 0.45 |
| Decision Tree (tuned) | 0.70 | 0.28 | 0.92 | 0.43 |
| **Decision Tree (tuned, 7 features, deployed)** | 0.70 | 0.28 | 0.91 | 0.43 |

## Hyperparameter Tuning

Tuning used `RandomizedSearchCV` with `cv=5`, `n_jobs=-1`, and `scoring='f1'` on the positive class, across two hyperparameters:

| Hyperparameter | Values searched | Best (DT) | Best (RF) |
|---|---|---|---|
| `class_weight` | None, 'balanced' | 'balanced' | 'balanced' |
| `max_depth` | 10, 20, None | 10 | 20 |

Balanced weighting made the 12% minority class count more heavily during training, which took Decision Tree recall from 0.30 up to 0.92.

## Final Model: Decision Tree

With the two effectively tied, deployment decided it. Both models rank customers equally well (AUC-PR 0.342 vs 0.343, about 2.8× better than ranking at random), so the difference in their reports comes down to where the 50% cut-off sits. The tuned Random Forest came out at roughly 120MB, over GitHub's 100MB file limit, while the tuned Decision Tree fits comfortably under. Since Random Forest wasn't buying any real performance, the Decision Tree was the simpler and lighter choice, and being one tree rather than a forest of them, it's also something you can actually walk a marketing stakeholder through.

## Feature Selection

`Vehicle_Damage` alone accounts for 79% of the model's decisions, with `Age` and `Previously_Insured` adding another 7% each, 93.9% between the three of them.

Given that, `Driving_License`, `Gender` and `Vehicle_Age` were dropped. Performance barely moved: F1 stayed the same and recall fell by 51 customers out of 14,013, which is a reasonable trade for a simpler model. It also means the model doesn't rely on gender to decide who gets targeted.

`Region_Code`, `Policy_Sales_Channel` and `Vintage` stayed in the model but were left out of the app's inputs. The first two are anonymised internal codes that no user could realistically answer, and together the three only make up about 4.7% of the model's decisions, so the app fills them with representative defaults instead: the median or most common value from the training data. That keeps the form down to four questions a user can actually answer, including the three features that drive most of the prediction.

## App Features

**Single Customer tab**

- Enter Age, Vehicle Damage history, existing insurance status and Annual Premium in the sidebar
- Returns an interest score and a contact recommendation, using a 50% threshold
- "Why this prediction" breaks down which factors pushed the score up or down
- "What would change this?" re-runs the model with the biggest factors flipped, so you can see how much each one is really doing

**Score a Customer List tab**

- Upload a CSV to score multiple customers at once
- Checks for missing columns, non-numeric values and invalid categories, with a clear message for each
- Downloadable example CSV so the expected format is obvious
- Results come back sorted by likelihood, highest first, with a recommended contact flag
- Download the scored list as a CSV, ready to hand to the call team

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Main Streamlit application |
| `ml_project.ipynb` | Full notebook: data understanding, preparation, modelling and evaluation |
| `insurance_final_model.pkl` | Trained Decision Tree model (loaded by the app) |
| `cars.png` | Header image displayed in the app |
| `demo_customers.csv` | Example customer list for testing the batch scoring feature |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Pins the Python version for Streamlit Cloud (the default version failed to install scipy) |

## Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`insurance_final_model.pkl` and `cars.png` need to be in the same folder as `streamlit_app.py`.

## Limitations

- It's a prioritisation tool, not a guarantee. Final outreach decisions should still be made by the marketing team.
- Predictions use 7 features but only 4 come from the user. The other three are held at fixed defaults, so accuracy drops for customers who sit far from those values.
- The model is tuned to favour recall over precision. That fits the business priority of not missing sales, but it does mean a fair share of contacted customers won't convert: about 1 in 3.5 is actually interested.
- The score is best read as a ranking, not a true probability. Balanced class weighting pushes scores upwards, so a customer shown at 67% is a strong prospect, but only about 28% of all flagged customers are actually interested.
- Scores come in steps. A decision tree gives every customer in the same leaf the same score, so different customers often end up with identical scores.
