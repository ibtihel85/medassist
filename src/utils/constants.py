"""
src/utils/constants.py
─────────────────────────────────────────────────────────────────────
Shared string constants and the quick-reference medical/statistical
terminology dictionary.
"""

# ── Answer modifiers ──────────────────────────────────────────────────

MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "⚕️  This information is for research purposes only and is NOT medical advice. "
    "Always consult a qualified healthcare professional."
)

LOW_CONF_PREFIX = (
    "⚠️  Low confidence — the retrieved abstracts only partially support "
    "this answer. Treat with caution.\n\n"
)

REFUSE_MSG = (
    "I was unable to find sufficiently relevant evidence in the database "
    "to answer this question reliably. "
    "Please try rephrasing, or consult a medical professional."
)

# ── Quick-reference medical/statistical terminology ───────────────────

QUICK_DEFINITIONS: dict[str, str] = {
    "rct":
        "Randomized Controlled Trial — participants randomly assigned to treatment "
        "or control; gold standard for establishing causality.",
    "cohort":
        "Group of patients followed over time in an observational study; "
        "shows association, not causation.",
    "meta-analysis":
        "Statistical pooling of multiple studies; increases statistical power. "
        "Watch for heterogeneity (I²).",
    "systematic review":
        "Comprehensive structured review of all evidence on a clinical question.",
    "placebo":
        "Inactive treatment given to control group to blind participants to "
        "treatment assignment.",
    "double blind":
        "Neither participants nor researchers know who receives treatment vs placebo.",
    "biomarker":
        "Measurable biological indicator of disease state, e.g. HbA1c for diabetes.",
    "comorbidity":
        "Presence of two or more chronic conditions in one patient simultaneously.",
    "efficacy":
        "Ability of a treatment to produce the desired effect under controlled conditions.",
    "prevalence":
        "Total existing cases of a disease in a population at a given time.",
    "incidence":
        "Number of new cases of a disease in a population over a defined period.",
    "mortality":
        "Death rate — often expressed as deaths per 100,000 population per year.",
    "morbidity":
        "Presence or degree of illness or disease in a population.",
    "pathogenesis":
        "The biological mechanism through which a disease develops.",
    "aetiology":
        "The cause or set of causes of a disease or condition.",
    "prognosis":
        "Predicted likely outcome or course of a disease.",
    "contraindication":
        "Condition or factor that makes a treatment inadvisable.",
    "pharmacokinetics":
        "How the body absorbs, distributes, metabolises, and excretes a drug (ADME).",
    "pharmacodynamics":
        "The effects a drug has on the body and its mechanism of action.",
    "p-value":
        "Probability of observing the result by chance if H0 is true. "
        "p < 0.05 is the conventional significance threshold.",
    "confidence interval":
        "Range of values consistent with the data. "
        "A 95% CI means 95% of such intervals contain the true value.",
    "odds ratio":
        "Ratio of odds of outcome in exposed vs unexposed group. OR > 1 = increased risk.",
    "hazard ratio":
        "Relative risk of an event occurring in one group vs another over time "
        "(used in survival analyses).",
    "number needed to treat":
        "NNT — number of patients that must be treated for one additional patient to benefit.",
    "sensitivity":
        "Test's ability to correctly identify true positives: TP / (TP + FN).",
    "specificity":
        "Test's ability to correctly identify true negatives: TN / (TN + FP).",
    "positive predictive value":
        "PPV — probability that a positive test result is a true positive.",
    "nnt":
        "Number Needed to Treat — patients that must be treated for one additional patient "
        "to benefit.",
    "ppv":
        "Positive Predictive Value — probability that a positive test result is a true positive.",
    "auc":
        "Area Under the ROC Curve — overall diagnostic accuracy of a test "
        "(0.5 = chance, 1.0 = perfect).",
    "roc":
        "Receiver Operating Characteristic — curve plotting sensitivity vs 1-specificity "
        "across all decision thresholds.",
}
