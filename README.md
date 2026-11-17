# Women-Governors-and-COVID-19-crisis-management-replication
This is a python replication of a study be Obenauer et al., 2024. Are women strategic leaders more effective during a crisis than men strategic leaders?


Contents
study_3A.py: U.S. county-border geographic matching: 
covariate balance and dyad fixed-effects regressions with state/border-clustered standard errors (Table 4 in the paper).
study_3B.py: Brazilian close-election sharp regression discontinuity: 
rdrobust models with and without controls (Table 5), 
covariate balance (Supplemental Table 17), 
bandwidth sensitivity (Supplemental Table 18), 
and deaths per 100,000 residents (one of the visual robustness controls). 
Includes figures and CSV exports.

*Additionally, simple OLS was estimated at the begining of the code, which was later replaced with rdrobust that authors use in the paper. 

Outputs (statistics tables and visual representations, graphs) are written to outputs (study 3B) and results (study 3A)
*** before tunning replace paths in the __need a path___ in both python files.

Results:
Study 3A (geographic matching): In both models results are not significant
Study 3B (RDD): In both models results are not significant
