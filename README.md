# DecodeLabs_Task01_UdamIlukpotha

# 🚢 Project 1: Advanced EDA & Feature Engineering
### Titanic Dataset | DecodeLabs Industrial Training — Batch 2026

***

## Project Overview

This project is **Project 1** of the DecodeLabs Data Science Industrial Training Kit (Batch 2026). The objective is to act as a real Data Scientist — not just loading data and running algorithms, but building a **production-grade preprocessing pipeline** that transforms raw, chaotic data into a mathematically clean dataset ready for machine learning models.

The dataset used is the **Titanic passenger dataset** (891 rows × 12 columns), chosen specifically because it contains all the real-world data challenges this project requires: missing values across multiple columns, outliers, categorical text fields, and rich enough features to engineer meaningful new variables.

***

## What This Project Covers

The pipeline follows the **Input → Process → Output (IPO) Architecture** described in the DecodeLabs training framework:

| Stage | Focus | Key Techniques |
|-------|-------|---------------|
| **Phase 1 — Input** | Securing data fidelity | Missing value detection, statistical imputation, KNN imputation |
| **Phase 2 — Process** | Transformations at scale | IQR outlier detection, Winsorization, One-Hot Encoding, multicollinearity eradication |
| **Phase 3 — Output** | Feature engineering | 5 new predictive features engineered from existing columns |

***

## Dataset

**File:** `titanic.csv`

| Column | Type | Description | Issues |
|--------|------|-------------|--------|
| `PassengerId` | int | Unique passenger ID | — |
| `Survived` | int | 0 = died, 1 = survived **(target variable)** | — |
| `Pclass` | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | — |
| `Name` | string | Passenger name | Dropped (non-informative) |
| `Sex` | categorical | male / female | Encoded |
| `Age` | float | Age in years | ~20% missing → KNN Imputed |
| `SibSp` | int | # siblings/spouses aboard | Outliers → Winsorized |
| `Parch` | int | # parents/children aboard | Outliers → Winsorized |
| `Ticket` | string | Ticket number | Dropped (non-informative) |
| `Fare` | float | Ticket price paid | Outliers → Winsorized |
| `Cabin` | string | Cabin number | ~77% missing → Dropped |
| `Embarked` | categorical | Port of embarkation (S/C/Q) | <1% missing → Rows dropped |

***

## Phase 1: Missing Value Handling

The **Missing Data Decision Matrix** was applied strictly — no guesswork:

| Column | Missingness | Rule Applied | Method |
|--------|-------------|--------------|--------|
| `Cabin` | ~77% | > 20% → Drop column | `df.drop(columns=['Cabin'])` |
| `Age` | ~20% | > 20% → Multi-dimensional estimation | `KNNImputer(n_neighbors=5)` |
| `Embarked` | < 1% | < 5% → Drop rows | `df.dropna(subset=['Embarked'])` |

**Why KNN for Age?** KNN Imputation uses neighboring features (`Pclass`, `SibSp`, `Parch`, `Fare`) to estimate missing Age values — capturing complex multi-dimensional relationships rather than simply filling with the column mean.

***

## Phase 2A: Outlier Detection & Neutralization (IQR)

Outliers were detected using the **Interquartile Range (IQR)** method:

```
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Columns checked: `Age`, `Fare`, `SibSp`, `Parch`

**Strategy: Winsorization (not deletion)**
Instead of dropping outlier rows (which destroys data volume), `numpy.clip()` was used to cap values exactly at the statistical boundaries. This preserves row count and sequential data integrity — critical for any temporal or sequential downstream modeling.

***

## Phase 2B: Categorical Encoding (One-Hot Encoding)

**Why not Label Encoding?**
Assigning ascending integers to nominal categories (e.g., S=1, C=2, Q=3) introduces a synthetic spatial hierarchy — it implies C is "twice S" and Q is "three times S". This is mathematically incorrect for nominal categories.

**One-Hot Encoding** maps each distinct category into its own orthogonal binary column. All categories become equidistant in coordinate space (√2 apart), which is the correct mathematical representation for nominal data.

Columns encoded:
- `Sex` → `Sex_male`
- `Embarked` → `Embarked_Q`, `Embarked_S`

***

## Phase 2C: Multicollinearity Detection

The correlation matrix was computed for all numeric features. Pairs with absolute correlation > 0.80 were flagged, and the column with the weaker relationship to the target (`Survived`) was dropped.

**Why this matters:** When predictor columns are highly correlated, the feature matrix **X** becomes singular (rank-deficient and non-invertible). Ordinary Least Squares coefficients become unstable — minor changes in training data cause large swings in predictions, destroying model generalization.

***

## Phase 3: Feature Engineering

5 new predictive features were engineered from the raw columns using domain knowledge about the Titanic disaster:

| Feature | Formula / Logic | Why It's Predictive |
|---------|----------------|---------------------|
| `FamilySize` | `SibSp + Parch + 1` | Larger families may have had different evacuation dynamics |
| `IsAlone` | `1 if FamilySize == 1 else 0` | Solo travelers had distinct survival patterns |
| `AgeGroup` | Binned: Child / Teenager / YoungAdult / Adult / Senior | Age stage captures evacuation priority better than raw age |
| `FareBand` | Quartile bins: Low / Medium / High / VeryHigh | Fare tiers correlate with class and deck location |
| `FamilyType` | Solo / Small (≤3) / Medium (≤5) / Large (>5) | Granular family dynamics beyond a simple size number |

All new categorical features were subsequently One-Hot Encoded.

***

## Final Dataset

**File:** `titanic_cleaned.csv`

| Property | Value |
|----------|-------|
| Rows | ~889 (after row drops) |
| Columns | ~20+ (after encoding and feature engineering) |
| Missing values | **0** |
| Target variable | `Survived` |
| Ready for ML | ✅ Yes |

***

## Notebook Structure

The full pipeline is in:
**`Titanic_Project1_EDA_FeatureEngineering.ipynb`**

| Cell Block | Content |
|-----------|---------|
| Step 0 | Library imports |
| Step 1 | Data loading & exploration (shape, dtypes, statistics) |
| Phase 1 | Missing value analysis → visualized → imputed |
| Phase 2A | IQR outlier detection → boxplots before/after → Winsorization |
| Phase 2B | Categorical encoding with explanation |
| Phase 2C | Correlation heatmap → multicollinearity eradication |
| Phase 3 | 5 new features engineered and encoded |
| Output | Final dataset summary + survival analysis charts + CSV export |

***

## Skills Demonstrated

- `pandas` — DataFrame manipulation, missing value handling, encoding, filtering
- `numpy` — Vectorized operations, `np.clip()` for Winsorization
- `sklearn.impute.KNNImputer` — Multi-dimensional missing value estimation
- `matplotlib` & `seaborn` — Boxplots, heatmaps, bar charts
- Statistical thinking — IQR, correlation matrices, imputation decision logic
- Feature extraction — Domain-driven feature creation

***

## Tools & Environment

- **Language:** Python 3
- **Environment:** VS Code with virtual environment (`venv`)
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

***

## Key Takeaway

> *"Machine learning estimators possess zero qualitative reasoning. They are numerical optimization algorithms operating on real-numbered coordinate spaces. If unrefined, low-fidelity data enters the system, the algorithm will flawlessly optimize for the wrong patterns. Data preprocessing is not janitorial work — it is the structural engineering of mathematical truth."*
>
> — DecodeLabs Industrial Training Kit, Batch 2026

***# 🚢 Project 1: Advanced EDA & Feature Engineering
### Titanic Dataset | DecodeLabs Industrial Training — Batch 2026

***

## Project Overview

This project is **Project 1** of the DecodeLabs Data Science Industrial Training Kit (Batch 2026). The objective is to act as a real Data Scientist — not just loading data and running algorithms, but building a **production-grade preprocessing pipeline** that transforms raw, chaotic data into a mathematically clean dataset ready for machine learning models.

The dataset used is the **Titanic passenger dataset** (891 rows × 12 columns), chosen specifically because it contains all the real-world data challenges this project requires: missing values across multiple columns, outliers, categorical text fields, and rich enough features to engineer meaningful new variables.

***

## What This Project Covers

The pipeline follows the **Input → Process → Output (IPO) Architecture** described in the DecodeLabs training framework:

| Stage | Focus | Key Techniques |
|-------|-------|---------------|
| **Phase 1 — Input** | Securing data fidelity | Missing value detection, statistical imputation, KNN imputation |
| **Phase 2 — Process** | Transformations at scale | IQR outlier detection, Winsorization, One-Hot Encoding, multicollinearity eradication |
| **Phase 3 — Output** | Feature engineering | 5 new predictive features engineered from existing columns |

***

## Dataset

**File:** `titanic.csv`

| Column | Type | Description | Issues |
|--------|------|-------------|--------|
| `PassengerId` | int | Unique passenger ID | — |
| `Survived` | int | 0 = died, 1 = survived **(target variable)** | — |
| `Pclass` | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | — |
| `Name` | string | Passenger name | Dropped (non-informative) |
| `Sex` | categorical | male / female | Encoded |
| `Age` | float | Age in years | ~20% missing → KNN Imputed |
| `SibSp` | int | # siblings/spouses aboard | Outliers → Winsorized |
| `Parch` | int | # parents/children aboard | Outliers → Winsorized |
| `Ticket` | string | Ticket number | Dropped (non-informative) |
| `Fare` | float | Ticket price paid | Outliers → Winsorized |
| `Cabin` | string | Cabin number | ~77% missing → Dropped |
| `Embarked` | categorical | Port of embarkation (S/C/Q) | <1% missing → Rows dropped |

***

## Phase 1: Missing Value Handling

The **Missing Data Decision Matrix** was applied strictly — no guesswork:

| Column | Missingness | Rule Applied | Method |
|--------|-------------|--------------|--------|
| `Cabin` | ~77% | > 20% → Drop column | `df.drop(columns=['Cabin'])` |
| `Age` | ~20% | > 20% → Multi-dimensional estimation | `KNNImputer(n_neighbors=5)` |
| `Embarked` | < 1% | < 5% → Drop rows | `df.dropna(subset=['Embarked'])` |

**Why KNN for Age?** KNN Imputation uses neighboring features (`Pclass`, `SibSp`, `Parch`, `Fare`) to estimate missing Age values — capturing complex multi-dimensional relationships rather than simply filling with the column mean.

***

## Phase 2A: Outlier Detection & Neutralization (IQR)

Outliers were detected using the **Interquartile Range (IQR)** method:

```
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Columns checked: `Age`, `Fare`, `SibSp`, `Parch`

**Strategy: Winsorization (not deletion)**
Instead of dropping outlier rows (which destroys data volume), `numpy.clip()` was used to cap values exactly at the statistical boundaries. This preserves row count and sequential data integrity — critical for any temporal or sequential downstream modeling.

***

## Phase 2B: Categorical Encoding (One-Hot Encoding)

**Why not Label Encoding?**
Assigning ascending integers to nominal categories (e.g., S=1, C=2, Q=3) introduces a synthetic spatial hierarchy — it implies C is "twice S" and Q is "three times S". This is mathematically incorrect for nominal categories.

**One-Hot Encoding** maps each distinct category into its own orthogonal binary column. All categories become equidistant in coordinate space (√2 apart), which is the correct mathematical representation for nominal data.

Columns encoded:
- `Sex` → `Sex_male`
- `Embarked` → `Embarked_Q`, `Embarked_S`

***

## Phase 2C: Multicollinearity Detection

The correlation matrix was computed for all numeric features. Pairs with absolute correlation > 0.80 were flagged, and the column with the weaker relationship to the target (`Survived`) was dropped.

**Why this matters:** When predictor columns are highly correlated, the feature matrix **X** becomes singular (rank-deficient and non-invertible). Ordinary Least Squares coefficients become unstable — minor changes in training data cause large swings in predictions, destroying model generalization.

***

## Phase 3: Feature Engineering

5 new predictive features were engineered from the raw columns using domain knowledge about the Titanic disaster:

| Feature | Formula / Logic | Why It's Predictive |
|---------|----------------|---------------------|
| `FamilySize` | `SibSp + Parch + 1` | Larger families may have had different evacuation dynamics |
| `IsAlone` | `1 if FamilySize == 1 else 0` | Solo travelers had distinct survival patterns |
| `AgeGroup` | Binned: Child / Teenager / YoungAdult / Adult / Senior | Age stage captures evacuation priority better than raw age |
| `FareBand` | Quartile bins: Low / Medium / High / VeryHigh | Fare tiers correlate with class and deck location |
| `FamilyType` | Solo / Small (≤3) / Medium (≤5) / Large (>5) | Granular family dynamics beyond a simple size number |

All new categorical features were subsequently One-Hot Encoded.

***

## Final Dataset

**File:** `titanic_cleaned.csv`

| Property | Value |
|----------|-------|
| Rows | ~889 (after row drops) |
| Columns | ~20+ (after encoding and feature engineering) |
| Missing values | **0** |
| Target variable | `Survived` |
| Ready for ML | ✅ Yes |

***

## Notebook Structure

The full pipeline is in:
**`Titanic_Project1_EDA_FeatureEngineering.ipynb`**

| Cell Block | Content |
|-----------|---------|
| Step 0 | Library imports |
| Step 1 | Data loading & exploration (shape, dtypes, statistics) |
| Phase 1 | Missing value analysis → visualized → imputed |
| Phase 2A | IQR outlier detection → boxplots before/after → Winsorization |
| Phase 2B | Categorical encoding with explanation |
| Phase 2C | Correlation heatmap → multicollinearity eradication |
| Phase 3 | 5 new features engineered and encoded |
| Output | Final dataset summary + survival analysis charts + CSV export |

***

## Skills Demonstrated

- `pandas` — DataFrame manipulation, missing value handling, encoding, filtering
- `numpy` — Vectorized operations, `np.clip()` for Winsorization
- `sklearn.impute.KNNImputer` — Multi-dimensional missing value estimation
- `matplotlib` & `seaborn` — Boxplots, heatmaps, bar charts
- Statistical thinking — IQR, correlation matrices, imputation decision logic
- Feature extraction — Domain-driven feature creation

***

## Tools & Environment

- **Language:** Python 3
- **Environment:** VS Code with virtual environment (`venv`)
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

***

## Key Takeaway

> *"Machine learning estimators possess zero qualitative reasoning. They are numerical optimization algorithms operating on real-numbered coordinate spaces. If unrefined, low-fidelity data enters the system, the algorithm will flawlessly optimize for the wrong patterns. Data preprocessing is not janitorial work — it is the structural engineering of mathematical truth."*
>
> — DecodeLabs Industrial Training Kit, Batch 2026

**
