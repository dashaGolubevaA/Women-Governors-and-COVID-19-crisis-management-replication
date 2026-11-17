
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
from rdrobust import rdrobust, rdplot, plot_rdrobust

# 1: check if data matches
#load dataset
ROOT = Path("__need a path___").resolve()
rd = pd.read_stata(ROOT / "study_3B_rdd.dta", convert_categoricals=False)

# only RD observations
rd = rd.loc[(rd.year == 2020) & rd.margin.notna()].copy() #this will be a full df
#treatment: female mayor
rd["female"] = rd["w_mayor"].astype(int)

# Interaction for separate slopes in the RD regression
rd["female_x_margin"] = rd["female"] * rd["margin"]

print(len(rd))
print(rd["female"].value_counts())

#2: select observations near the cutoff

h = 0.212
df = rd.loc[rd["margin"].abs() <= h].copy() #data frame that is close to the cut-off with which we will be working.

print("Full RD sample:", len(rd))
print("Within bandwidth:", len(df))

# Check treatment-group sizes
print(df["female"].value_counts())

# 3: Calculate triangular kernel weights

df["weight"] = (1 - df["margin"].abs() / h)
print(df[["margin", "female", "weight"]].head())


# 4: Weighted least squares regression
model = smf.wls(formula="deaths ~ female + margin + female_x_margin", 
                data=df, weights=df["weight"]).fit(cov_type="cluster",
                cov_kwds={"groups": df["codeuf"]})

print(model.summary())

# 5: visuals

fig, ax = plt.subplots(figsize=(9, 5))

# Plot binned observations on each side
WHARTON_BLUE = "#011F5B"
WHARTON_RED = "#990000"
for gender, color, label in [(0, WHARTON_BLUE, "Male mayor"), (1, WHARTON_RED, "Female mayor")]:
    subset = df[df["female"] == gender]
    edges = (np.linspace(-h, 0, 41)
        if gender == 0
        else np.linspace(0, h, 41))

    subset = subset.copy()
    subset["bin"] = pd.cut(subset["margin"], edges, include_lowest=True)

    means = subset.groupby("bin", observed=True)[["margin", "deaths"]].mean()
    #ax.scatter(subset["margin"],
    #subset["deaths"], color=color, alpha=0.25, s=12, label=label)
    ax.scatter(means["margin"], means["deaths"], color=color, label=label, s=10)

# Plot regression lines
for gender, color, x_min, x_max in [(0, WHARTON_BLUE, -h, 0), (1, WHARTON_RED, 0, h)]:
    x = np.linspace(x_min, x_max, 100)

    prediction_data = pd.DataFrame({"female": gender, "margin": x, "female_x_margin": gender * x})
    
    #fitted values and 95% confidence intervals
    predictions = model.get_prediction(prediction_data).summary_frame(alpha=0.05)
    ax.plot(x, predictions["mean"], color=color, linewidth=2)

    # CIs
    ax.fill_between(x, predictions["mean_ci_lower"].to_numpy(), predictions["mean_ci_upper"].to_numpy(), color=color, alpha=0.15)


#cutoff
ax.axvline(0, color="black", linestyle="--")

ax.set_xlabel("Female candidate's electoral margin")
ax.set_ylabel("COVID-19 deaths")
ax.set_title("RDD: Female Mayors and COVID-19 Deaths")

ax.legend()
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "rdd_figure.png", dpi=300)
plt.show()



# Save
#regression output
with open(ROOT / "outputs" / "model1_summary.txt", "w") as f:
    f.write(model.summary().as_text())

# coefficient table with ci
results = pd.DataFrame({"coefficient": model.params, "std_error": model.bse, "p_value": model.pvalues, "CI_lower": model.conf_int()[0], "CI_upper": model.conf_int()[1]})

results.to_csv(ROOT / "outputs" / "model1_results.csv")
print(results.round(4))


## Model 2:

controls = "age + d_PP + d_PMDB + d_PSDB + d_PSD + reeleito + pop19"

model2 = smf.wls(formula="deaths ~ female + margin + female_x_margin + " + controls, data=df, weights=df["weight"]).fit(cov_type="cluster", cov_kwds={"groups": df["codeuf"]})

print(model2.summary())


# 8: Table 5 replication, using rdrobust (the estimator that was used in the paper)

# Model 1: Without controls
rd_model1 = rdrobust(y=rd["deaths"], x=rd["margin"], c=0, p=1, q=2, kernel="tri", bwselect="mserd", vce="nn", cluster=rd["codeuf"])

# Model 2: With controls
controls = ["age", "d_PP", "d_PMDB", "d_PSDB", "d_PSD", "reeleito", "pop19"]

rd_model2 = rdrobust(y=rd["deaths"], x=rd["margin"], covs=rd[controls], c=0, p=1, q=2,
                      kernel="tri", bwselect="mserd", vce="nn", cluster=rd["codeuf"])

print("MODEL 1, rdrobust")
print(rd_model1)

print("MODEL 2, rdrobust")
print(rd_model2)


# Save 
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

with open(ROOT / "outputs" / "table5_rdrobust.txt", "w") as f:
    f.write("MODEL 1\n" + str(rd_model1))
    f.write("\n\nMODEL 2\n" + str(rd_model2))

#results
results = []

for name, m in [("Model 1", rd_model1),
    ("Model 2", rd_model2)]:
    results.append({"Model": name,"Conventional estimate": m.Estimate["tau.us"].iloc[0],"Bias-corrected estimate": m.Estimate["tau.bc"].iloc[0], 
                    "Conventional SE": m.Estimate["se.us"].iloc[0], "Robust SE": m.Estimate["se.rb"].iloc[0], "Effective N": sum(m.N_h),"Total N": sum(m.N),"CI lower": m.ci.iloc[2, 0],"CI upper": m.ci.iloc[2, 1]})

table5 = pd.DataFrame(results)
print(table5.round(3).to_string(index=False))
table5.to_csv(ROOT / "outputs" / "table5_rdrobust.csv", index=False)




# 9: Table 5 RD plots - actual rdrobust bandwidths, Penn colors

for name, est, covariates in [("model1", rd_model1, None), ("model2", rd_model2, rd[controls])]:
    h_left, h_right = est.bws.loc["h", ["left", "right"]].to_numpy(dtype=float)
    mask = rd["margin"].between(-h_left, h_right)

    plot = rdplot(y=rd.loc[mask, "deaths"], x=rd.loc[mask, "margin"], covs=covariates.loc[mask] if covariates is not None else None, p=1, kernel="tri", h=[h_left, h_right], nbins=[25, 25], ci=95, hide=True)

    bins = plot.vars_bins
    fits = plot.vars_poly

    fig, ax = plt.subplots(figsize=(9, 5))

    for left, color, label in [(True, WHARTON_BLUE, "Male mayor"), (False, WHARTON_RED, "Female mayor")]:
        b = bins[bins["rdplot_mean_bin"] < 0] if left else bins[bins["rdplot_mean_bin"] >= 0]
        f = fits[fits["rdplot_x"] < 0] if left else fits[fits["rdplot_x"] >= 0]
        f = f.sort_values("rdplot_x")

        ax.scatter(b["rdplot_mean_bin"], b["rdplot_mean_y"], color=color, s=20, label=label)
        ax.plot(f["rdplot_x"], f["rdplot_y"], color=color, linewidth=2.5)

        ax.vlines(b["rdplot_mean_bin"], b["rdplot_ci_l"], b["rdplot_ci_r"], color=color, alpha=0.25, linewidth=1)

    ci_low, ci_high = est.ci.iloc[2, 0], est.ci.iloc[2, 1]
    ax.text(0.98, 0.97, f"Robust 95% CI: [{ci_low:.2f}, {ci_high:.2f}]", transform=ax.transAxes, ha="right", va="top")

    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Female candidate's electoral margin")
    ax.set_ylabel("COVID-19 deaths")
    ax.set_title(f"RDD: Female Mayors and COVID-19 Deaths — {name.upper()}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUTS / f"table5_{name}_rdplot.png", dpi=300)
    plt.show()
    plt.close(fig)



# 10: Recreate Table 5

models = [rd_model1, rd_model2]

table5 = pd.DataFrame({"Model 1": [
        f'{rd_model1.Estimate["tau.bc"].iloc[0]:.2f}',
        f'({rd_model1.Estimate["se.rb"].iloc[0]:.2f})',
        "No", sum(rd_model1.N_h), sum(rd_model1.N)], 
        "Model 2": [
        f'{rd_model2.Estimate["tau.bc"].iloc[0]:.2f}',
        f'({rd_model2.Estimate["se.rb"].iloc[0]:.2f})',
        "Yes", sum(rd_model2.N_h), sum(rd_model2.N)]
}, index=["Mayor gender (female = 1)", "Standard error", "Control variables included", "Number of effective observations", "Number of observations"])

print(table5)
table5.to_csv(ROOT / "outputs" / "table5.csv")

# Save as a formatted image
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.axis("off")
ax.set_title("Table 5. Results of the Non-Parametric Sharp RD Analysis (Study 3B)", fontsize=12, fontweight="bold", pad=20)

tbl = ax.table(cellText=table5.values, rowLabels=table5.index, colLabels=table5.columns, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.7)

plt.savefig(ROOT / "outputs" / "table5.png", dpi=300, bbox_inches="tight")
plt.show()



# 11: Supplemental Table 1. Covariate balance

balance_results = []

for variable in controls:
    m = rdrobust(y=rd[variable], x=rd["margin"], c=0, p=1, q=2, kernel="tri", bwselect="mserd", vce="nn", cluster=rd["codeuf"])

    balance_results.append({"Variable": variable, "Coefficient": m.Estimate["tau.bc"].iloc[0], "SE": m.Estimate["se.rb"].iloc[0], "p-value": m.pv.iloc[2, 0], "CI lower": m.ci.iloc[2, 0], "CI upper": m.ci.iloc[2, 1]})

table17 = pd.DataFrame(balance_results)
table17 = table17.round(3)
print(table17.to_string(index=False))
table17.to_csv(ROOT / "outputs" / "supplemental_table17_cov_balance.csv", index=False)


#brandwidth sensetivity 

# 12: Supplemental Table 18 - Bandwidth sensitivity

bandwidth_results = []

for h_test in [0.025, 0.05, 0.10, 0.15]: #different thresholds
    for name, covariates in [("No", None), ("Yes", rd[controls])]:
        m = rdrobust(y=rd["deaths"], x=rd["margin"], covs=covariates, c=0, h=h_test, p=1, q=2, kernel="tri", vce="nn", cluster=rd["codeuf"])

        bandwidth_results.append({"Bandwidth": h_test, "Controls": name, "Coefficient": m.Estimate["tau.bc"].iloc[0], "SE": m.Estimate["se.rb"].iloc[0], "p-value": m.pv.iloc[2, 0],
        "CI lower": m.ci.iloc[2, 0], "CI upper": m.ci.iloc[2, 1], "Effective N": sum(m.N_h)})

table18 = pd.DataFrame(bandwidth_results)
table18 = table18.round(3)
print(table18.round(3).to_string(index=False))
table18.to_csv(ROOT / "outputs" / "supplemental_table18.csv", index=False)


#  save and plot

display = pd.DataFrame({f"Model {i+1}": [f"{r['Bandwidth']:.3f}", f"{r['Coefficient']:.3f}", f"({r['SE']:.3f})", r["Controls"]]
    for i, r in table18.iterrows()}, index=["Bandwidth", "Mayor gender (female = 1)", "Standard error", "Controls included"])

fig, ax = plt.subplots(figsize=(16, 4))
ax.axis("off")
ax.set_title("Supplemental Table 18: RD Bandwidth Sensitivity", fontsize=14, fontweight="bold", pad=20)

tbl = ax.table(cellText=display.values, rowLabels=display.index, colLabels=display.columns, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)

#pretty plot
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#011F5B")
        cell.set_text_props(color="white", weight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#F0F2F5")

plt.savefig(ROOT / "outputs"  / "supplemental_table18.png", dpi=300, bbox_inches="tight")
plt.show()

# 13: deaths per 100,000 inhabitants
rd["deaths_per_100k"] = np.where(rd["pop19"] > 0, rd["deaths"] / rd["pop19"] * 100000, np.nan)
rd_rate = rd.loc[rd["deaths_per_100k"].notna()].copy()
rate_model1 = rdrobust(y=rd_rate["deaths_per_100k"], x=rd_rate["margin"], c=0, p=1, q=2, kernel="tri", bwselect="mserd", vce="nn", cluster=rd_rate["codeuf"])

rate_model2 = rdrobust(y=rd_rate["deaths_per_100k"], x=rd_rate["margin"], covs=rd_rate[controls], c=0, p=1, q=2, kernel="tri", bwselect="mserd", vce="nn", cluster=rd_rate["codeuf"])
fig_rate1 = plot_rdrobust(rate_model1, y=rd_rate["deaths_per_100k"], x=rd_rate["margin"], show_effect=True)
fig_rate1.savefig(ROOT / "outputs" / "rate_model1.png", dpi=300, bbox_inches="tight")
plt.show()

# 14: Deaths per 100,000, RD graphs

WHARTON_BLUE = "#011F5B"
WHARTON_RED = "#990000"

for name, est, adjusted in [("model1", rate_model1, False), ("model2", rate_model2, True)]:
    h_left, h_right = est.bws.loc["h", ["left", "right"]].to_numpy(dtype=float)
    mask = rd_rate["margin"].between(-h_left, h_right)

    plot = rdplot(y=rd_rate.loc[mask, "deaths_per_100k"], x=rd_rate.loc[mask, "margin"], covs=rd_rate.loc[mask, controls] if adjusted else None, p=1, kernel="tri", h=[h_left, h_right], nbins=[25, 25], ci=95, hide=True)

    bins = plot.vars_bins
    fits = plot.vars_poly

    fig, ax = plt.subplots(figsize=(9, 5))

    for left, color, label in [(True, WHARTON_BLUE, "Male mayor"), (False, WHARTON_RED, "Female mayor")]:
        b = bins[bins["rdplot_mean_bin"] < 0] if left else bins[bins["rdplot_mean_bin"] >= 0]
        f = fits[fits["rdplot_x"] < 0] if left else fits[fits["rdplot_x"] >= 0]
        f = f.sort_values("rdplot_x")

        ax.scatter(b["rdplot_mean_bin"], b["rdplot_mean_y"], color=color, s=25, label=label)
        ax.plot(f["rdplot_x"], f["rdplot_y"], color=color, linewidth=2.5)
        ax.vlines(b["rdplot_mean_bin"], b["rdplot_ci_l"], b["rdplot_ci_r"], color=color, alpha=0.25)

    effect = est.Estimate["tau.bc"].iloc[0]
    ci_low, ci_high = est.ci.iloc[2, 0], est.ci.iloc[2, 1]

    ax.axvline(0, color="black", linestyle="--")
    ax.text(0.98, 0.97, f"RD = {effect:.2f}\nRobust 95% CI: [{ci_low:.2f}, {ci_high:.2f}]", transform=ax.transAxes, ha="right", va="top")
    ax.set_xlabel("Female candidate's electoral margin")
    ax.set_ylabel("COVID-19 deaths per 100,000 inhabitants")
    ax.set_title(f"RDD: COVID-19 Deaths per 100,000 — {name.upper()}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUTS / f"deaths_100k_{name}.png", dpi=300)
    plt.show()



# 15: Covariate balance forest plot - Supplemental Table 17

labels = {"age": "Mayor age", "pop19": "Population (2019)", "d_PP": "PP party", "d_PMDB": "PMDB party", "d_PSDB": "PSDB party", "d_PSD": "PSD party", "reeleito": "Reelected mayor"}

plot_df = table17.copy()
plot_df["SD"] = plot_df["Variable"].map({v: rd[v].std() for v in labels})
plot_df["Estimate"] = plot_df["Coefficient"] / plot_df["SD"]
plot_df["Lower"] = plot_df["CI lower"] / plot_df["SD"]
plot_df["Upper"] = plot_df["CI upper"] / plot_df["SD"]
plot_df["Label"] = plot_df["Variable"].map(labels)
plot_df = plot_df.iloc[::-1]

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(plot_df))

ax.errorbar(plot_df["Estimate"], y, xerr=[plot_df["Estimate"] - plot_df["Lower"], plot_df["Upper"] - plot_df["Estimate"]], fmt="o", color="#011F5B", ecolor="#011F5B", markersize=8, capsize=4, linewidth=2)

ax.axvline(0, color="#990000", linestyle="--", linewidth=2)
ax.set_yticks(y, plot_df["Label"])
ax.set_xlabel("Standardized RD estimate (95% confidence interval)")
ax.set_title("Covariate Balance Around the Electoral Cutoff", fontsize=14, fontweight="bold", pad=18)

ax.grid(axis="x", alpha=0.15)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)

plt.tight_layout()
plt.savefig(OUTPUTS / "covariate_balance_forest.png", dpi=300, bbox_inches="tight")
plt.show()
