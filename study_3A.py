
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from pathlib import Path
import matplotlib.pyplot as plt

# Save all results 
OUTPUT = Path("__need a path___").resolve().parent / "results"
OUTPUT.mkdir(exist_ok=True)

#function for saving
def save_table(table, name):
    table.to_csv(OUTPUT / f"{name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, max(2, len(table) * 0.38 + 1)))
    ax.axis("off")
    plot = ax.table(cellText=table.fillna("").values, colLabels=table.columns, colWidths=[0.60, 0.20, 0.20], loc="center", cellLoc="center")
    plot.auto_set_font_size(False)
    plot.set_fontsize(9)
    plot.scale(1, 1.3)
    fig.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

df = pd.read_stata("study_3A_geographic_matching.dta")

# variables construction
df["gov_gender"] = (df["gov_gender"] == "female").astype(int)
df["gov_party"] = (df["gov_party"] == "democrat").astype(int)
df["population_100k"] = df["tot_pop0"] / 100000
df["age60plus"] = 100 * df[[f"tot_pop{i}" for i in range(13, 19)]].sum(axis=1) / df["tot_pop0"] #they store their data in groups of people. grous that are higher than 60 start from 13

# covariate balance
county = df.drop_duplicates("countyfips")
controls = ["gov_termtenure", "gov_age", "gov_party", "size_sqmiles", "population_100k", "age60plus"]

women, men = county[county.gov_gender == 1], county[county.gov_gender == 0]
balance = pd.DataFrame([{"variable": c, "SMD": (women[c].mean() - men[c].mean()) / ((women[c].var() + men[c].var()) / 2)**0.5, "variance_ratio": women[c].var() / men[c].var()} for c in controls])

save_table(balance.round(3), "covariate_balance")

# Model 1: Dyad fixed-effects OLS
y = df["covid_death_5may2020"].astype(float)
X = df[["gov_gender"]].astype(float)

y_fe = y - y.groupby(df["dyad"], observed=True).transform("mean")
X_fe = X - X.groupby(df["dyad"], observed=True).transform("mean")

model1 = sm.OLS(y_fe, X_fe).fit()

# standard errors
state = df["statefips"].astype("category").cat.codes.to_numpy()
border = df["state_border"].astype("category").cat.codes.to_numpy()

cov, _, _ = cov_cluster_2groups(model1, state, border, use_correction=False)
n, k, G = len(df), X.shape[1], min(len(np.unique(state)), len(np.unique(border)))
cov *= (n - 1) / (n - k - 1) * G / (G - 1)

coef = model1.params["gov_gender"]
se = np.sqrt(cov[0, 0])
p = 2 * t.sf(abs(coef / se), df=G - 1)
r2 = 1 - sum(model1.resid**2) / sum((y - y.mean())**2)

# Save 
table4 = pd.DataFrame({
    "Variable": ["Governor gender (1 = woman)", "", "Governor term tenure", "", "Governor age", "", "Governor party (1 = democrat)", "", "County size", "", "Total county population", "", "Percentage of population 60 years and older", "", "Dyad fixed-effects included", "R²"],
    "Model 1": [f"{coef:.2f}", f"({se:.2f})"] + [""] * 12 + ["Yes", f"{r2:.3f}"],
    "Model 2": [""] * 16})

save_table(table4, "study3a_table4")



# Model 2: OLS with six controls
X2 = df[["gov_gender"] + controls].astype(float)
X2_fe = X2 - X2.groupby(df["dyad"], observed=True).transform("mean")

model2 = sm.OLS(y_fe, X2_fe).fit()

# Two-way clustered standard errors
cov2, _, _ = cov_cluster_2groups(model2, state, border, use_correction=False)
k2 = X2.shape[1]
cov2 *= (n - 1) / (n - k2 - 1) * G / (G - 1)

se2 = pd.Series(np.sqrt(np.diag(cov2)), index=X2.columns)
r2_2 = 1 - np.sum(model2.resid**2) / np.sum((y - y.mean())**2)

# Complete Table 4 and save both models
table4["Model 2"] = [v for c in X2.columns for v in (f"{model2.params[c]:.2f}", f"({se2[c]:.2f})")] + ["Yes", f"{r2_2:.3f}"]

save_table(table4, "study3a_table4")

#clean covariate balance


# Format covariate balance table
labels = {
    "gov_termtenure": "Governor tenure (years)",
    "gov_age": "Governor age (years)",
    "gov_party": "Governor party (1 = Democrat)",
    "size_sqmiles": "County size (square miles)",
    "population_100k": "County population",
    "age60plus": "Population aged 60+ (%)"}

balance_pretty = balance.rename(columns={"variable": "Covariate", "SMD": "Standardized Mean Difference", "variance_ratio": "Variance Ratio"}).copy()
balance_pretty["Covariate"] = balance_pretty["Covariate"].replace(labels)
balance_pretty.iloc[:, 1:] = balance_pretty.iloc[:, 1:].round(3)

balance_pretty.to_csv(OUTPUT / "covariate_balance.csv", index=False)
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.axis("off")

ax.text(0, 0.98, "Table 1. Covariate Balance Between Counties in Woman- and Man-Governed States", fontsize=13, fontweight="bold", transform=ax.transAxes)

tbl = ax.table(cellText=balance_pretty.values, colLabels=balance_pretty.columns, colWidths=[0.48, 0.32, 0.20], cellLoc="center", loc="center", bbox=[0, 0.27, 1, 0.55])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

for (r, c), cell in tbl.get_celld().items():
    cell.set_linewidth(0)
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.visible_edges = "BT"
        cell.set_linewidth(1)
    elif r == len(balance_pretty):
        cell.visible_edges = "B"
        cell.set_linewidth(1)
    if c == 0:
        cell.get_text().set_ha("left")

fig.savefig(OUTPUT / "covariate_balance.png", dpi=300, bbox_inches="tight")
plt.close(fig)
