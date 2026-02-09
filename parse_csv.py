import pandas as pd
import re

# Read compare_lp_formulation_results.csv
df = pd.read_csv("compare_lp_formulation_results.csv")

# For all entries
# Extract alpha values
for i in range(len(df)):
    lp_result = df.loc[i, "optimal_alpha_lp"]
    simple_lp_result = df.loc[i, "optimal_alpha_simple_lp"]

    lp_alpha = re.search(r"'alpha': ([0-9.]+)", lp_result)
    simple_lp_alpha = re.search(r"'alpha': ([0-9.]+)", simple_lp_result)

    # Store lp_alpha and simple_lp_alpha back to dataframe
    df.at[i, "optimal_alpha_lp"] = float(lp_alpha.group(1)) if lp_alpha else None
    df.at[i, "optimal_alpha_simple_lp"] = float(simple_lp_alpha.group(1)) if simple_lp_alpha else None
    

# Save the results to a new CSV file
results_df = df[["u_arrival_time", "optimal_alpha_lp", "optimal_alpha_simple_lp"]]
results_df.to_csv("compare_lp_formulation_alpha.csv", index=False)
