# Vehicle Insurance Cross-Sell Predictor 🚗

A Streamlit app that predicts whether an existing health insurance customer is likely to be interested in vehicle insurance, so marketing teams can focus outreach on the customers most likely to say yes.

**Live app:** (https://vehicle-insurance-prediction-6xip7eynikmvuh6x4o86ky.streamlit.app/)

## Business Problem

The company already has 381,109 health insurance customers and wants to cross-sell vehicle insurance to them. Calling all of them is expensive and time-consuming, and most of the people picking up the phone might not be interested anyway. Only around 12% are.

So the real problem is not reaching customers, it's about working out which ones are worth reaching. This app scores every customer on how likely they are to say yes and purchase vehicle insurance, which lets the marketing team work down a ranked list instead of blanket-calling the whole base of customers. In practice that cuts the list by about 60% while still reaching roughly 9 in 10 of the people who would have converted.

## Dataset

- **Source**: [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) (Kaggle)
- **Size**: 381,109 rows, no missing values or duplicates
- **Target**: `Response` — heavily imbalanced at roughly 12% interested to 88% not

That imbalance shapes everything downstream. A model that simply predicts "not interested" every time scores 88% accuracy while being completely useless, so accuracy was ruled out as a headline metric from the start.

Only `train.csv` was used — Kaggle's `test.csv` has no target column, so there's nothing to evaluate predictions against.

## Modelling Approach

Three models were compared: Decision Tree, Random Forest, and Gradient Boosted Trees.

The data was split 70/30 with `random_state=2026`. `Region_Code` and `Policy_Sales_Channel` both have high cardinality, so each was cut down to its top 10 categories with everything else grouped as `Other`. Those thresholds were worked out from the training set alone, so no test set information leaks back into preprocessing.

Class weighting made the biggest difference by far. With no weighting, one Decision Tree config scored F1 = 0.013. But with `class_weight='balanced'` at the same depth, it scored 0.428.

The Gradient Boosted Tree is the clearest illustration of why accuracy was the wrong metric here. It posted the highest accuracy of any model at 0.88, and found almost no interested customers at all (F1 ≈ 0.00 on the positive class). It got that score by predicting "not interested" for nearly everyone.

This left Decision Tree and Random Forest, which finished close on F1 for the interested class (0.43 vs 0.45) but pulled in different directions:

- **Decision Tree** — higher recall at 0.92, so it catches more real prospects but flags more people who won't convert
- **Random Forest** — higher precision at 0.31 vs 0.28, so fewer false alarms but more missed buyers

### Hyperparameter Tuning

Tuning used `RandomizedSearchCV` with `cv=5`, `n_jobs=-1`, and `scoring='f1'` on the positive class, across two hyperparameters:

| Hyperparameter | Values searched | Best (DT) | Best (RF) |
|---|---|---|---|
| `class_weight` | `None`, `'balanced'` | `'balanced'` | `'balanced'` |
| `max_depth` | `10`, `20`, `None` | `10` | `20` |

Balanced weighting was what made the 12% minority class count more heavily during training took Decision Tree recall from 0.30 up to 0.92.

### Final Model: Decision Tree

With F1 tied, deployment decided it. The tuned Random Forest came out at roughly 120MB, over GitHub's 100MB file limit, while the tuned Decision Tree fits comfortably under. Since Random Forest wasn't buying any real performance, the Decision Tree was the simpler and lighter choice, and being one tree rather than a forest of them, it's also something you can actually walk a marketing stakeholder through.

### Feature Selection

`Vehicle_Damage` alone accounts for 79% of the model's decisions, with `Age` and `Previously_Insured` adding another 7% each — 93.9% between the three of them.

Given that, `Driving_License`, `Gender` and `Vehicle_Age` were dropped. Performance barely moved: F1 stayed the same and recall fell by 51 customers out of 14,013, which is a reasonable trade for a simpler model.

`Region_Code`, `Policy_Sales_Channel` and `Vintage` stayed in the model but were left out of the app's inputs. They're anonymised internal codes that no user could realistically answer, and they only make up about 4.7% of the model's decisions, so the app fills them with representative defaults instead — the median or most common value from the training data. That keeps the form focused on the four things that actually drive the prediction.

## App Features

**Single Customer tab**
- Enter Age, Vehicle Damage history, existing insurance status and Annual Premium in the sidebar
- Returns a predicted probability of interest and a contact recommendation, using a 50% threshold
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
| `insurance_final_model.pkl` | Trained Decision Tree model (loaded by the app) |
| `cars.png` | Header image displayed in the app |
| `demo_customers.csv` | Example customer list for testing the batch scoring feature |
| `requirements.txt` | Python dependencies |

## Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`insurance_final_model.pkl` and `cars.png` need to be in the same folder as `streamlit_app.py`.

## Limitations

- It's a prioritisation tool, not a guarantee. Final outreach decisions should still be made by the marketing team.
- Predictions use 7 features but only 4 come from the user. The other three are held at fixed defaults, so accuracy drops for customers who sit far from those values.
- The 50% threshold favours recall over precision. That fits the business priority of not missing sales, but it does mean a fair share of contacted customers won't convert.
