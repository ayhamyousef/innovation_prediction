# Learning and predicting patent technology reuse trajectories

Code and derived results for a study of how the long-term reuse of a newly emerged patent
technology relates to information observable at the moment it emerges.

A technology is a pairing of two six-digit IPC codes. A technology is novel in the year that
pairing first appears on a patent. Its reuse trajectory is the cumulative count of later
inventions that repeat the pairing, over a twenty-year window. From USPTO records for 2002 to
2022 we assemble 201,710 novel technologies that accumulate at least twenty reuses, and
describe each by seven features computed in its emergence year.

No reuse-pattern labels exist in advance, so they have to be constructed, and this study treats
that construction as the substantive question rather than as a preliminary. Labels are built
along two separate routes.

The **sequence-based route** clusters the reuse trajectories themselves, in the latent space of
a GRU autoencoder trained to reconstruct them. Labels from this route depend only on how reuse
unfolded and on no hand-crafted quantity, so predicting them from emergence-time features is a
prediction problem with a genuinely external target.

The **emergence-profile route** selects among the seven features with a Fractal Autoencoder and
clusters the selected features directly. Labels from this route describe a technology at birth.
Recovering them from those same features is a consistency check rather than a forecast, and the
near-perfect accuracy it produces should be read as such.

The two partitions agree only marginally above chance, at an Adjusted Rand Index near 0.04.
That disagreement is itself one of the results: a partition of emergence-time covariates is not
a reuse-pattern taxonomy, although the two are often treated as interchangeable.

## Repository layout

    config/         pipeline configuration
    src/
      data/         Open Data Portal client, technology extraction, feature computation
      models/       FAE, the GRU sequence autoencoder, and the seven tabular classifiers
      utils/        shared helpers
    scripts/        numbered pipeline stages and analyses, run in order
    results/        derived metrics, summary tables and figures

## Requirements

Python 3.10 or later. Install dependencies with `pip install -r requirements.txt`. A
`Dockerfile` and `docker-compose.yml` are provided for a pinned environment.

Fetching patent records requires a USPTO Open Data Portal API key, supplied through the
environment:

    export ODP_API_KEY=your_key_here

The key is read from the environment. It is never written to disk or to configuration.

## Reproducing the study

Raw patent records are not versioned here. They are public, and stage 01 retrieves them.
Expect the fetch to take several hours and to produce several gigabytes.

    python scripts/01_fetch_data.py --start-year 2002 --end-year 2022
    python scripts/02_build_trajectories.py     # technologies, trajectories, seven features
    python scripts/03_cluster_patterns.py       # trajectory-shape labels
    python scripts/04_feature_selection.py      # FAE selection, K in {3,4,5}
    python scripts/04b_recluster.py             # emergence-profile labels
    python scripts/05_classify.py               # the seven tabular classifiers
    python scripts/06_ablation_subsets.py       # exhaustive feature-subset ablation
    python scripts/16_sequence_autoencoder.py   # GRU autoencoder, sequence-based labels
    python scripts/17_gru_ksweep_downstream.py  # cluster-count sweep and downstream prediction

Stages 07 through 15 produce the supporting analyses and the figures. Stages 18, 21 and 23
through 26 are the analyses added during revision, described in the next section.
`run_experiments.py` orchestrates the full classification matrix.

Every stage takes its seed from the configuration and defaults to 42. Feature selection and
clustering are fitted on the full corpus, because they constitute label construction; the
stratified 60/40 train and test split is applied only at the classification stage.

## Analyses added during revision

These back specific claims and can be run once the pipeline above has completed.

`scripts/23_gru_per_class_metrics.py` reports per-class precision, recall, F1, ROC-AUC and
average precision for the sequence-based labels. It establishes that the smallest of the three
classes is the best ranked despite having the lowest F1, so its low F1 is a decision-threshold
effect rather than a limit on identifying the class.

`scripts/24_feature_truncation_audit.py` quantifies a defect in the corpus. `ACCESS_SIZE` and
`ACCESS_TREND` are computed over the five years preceding emergence, but the patent record
assembled here begins in 2002. For technologies emerging in 2002 that window falls entirely
outside the data and `ACCESS_SIZE` is zero for all 26,471 of them; the feature stays
undercounted until the 2007 cohort, the first with a fully covered window. Affected cohorts
total 137,118 of 201,710 technologies. The script reports what this does to the cluster
contrasts, with and without the affected cohorts.

`scripts/25_recency_shortcut_check.py` asks how much of the predictability of the
sequence-based labels follows from emergence cohort rather than from the features. It compares
the seven features against emergence year alone, and against the feature set with the two
truncated features removed.

`scripts/26_censoring_free_robustness.py` removes right-censoring rather than measuring it.
Restricting to technologies that emerged in 2012 or earlier and truncating every trajectory to
ten years leaves a subcohort in which all 179,459 members are observed for exactly the same
span. Setting `MAX_COHORT=2022` includes the censored cohorts at the same horizon, which
isolates the effect of censoring from the effect of the horizon itself.

## Known limitations of the data

Two are worth stating plainly for anyone building on this.

The corpus is right-censored. Only 34.2 percent of technologies are observed for the full
twenty-year window, and trajectories of later cohorts are flat beyond the 2022 horizon by
construction. Within the window, a delayed take-off cannot be distinguished from censoring.

Two of the seven features are left-truncated for the earliest cohorts, for the reason given
above. Because `ACCESS_SIZE` is one of the three features the Fractal Autoencoder selects, part
of what the emergence-profile partition separates is emergence cohort rather than technological
position. Records extending before 2002 and beyond 2022 would be needed to disentangle the two
effects fully.

## What is not in this repository

Raw and cached patent records, which are public and are retrieved by stage 01. Per-technology
labeled tables, which run to tens of megabytes each and are regenerated by the scripts that
consume them. The manuscript source, because the paper is under review.

## The seven features

Computed in the emergence year, following Chen et al. (2025).

| Feature | Description |
|---|---|
| `ACCESS_SIZE` | Patents on either component over the five years before emergence |
| `ACCESS_TREND` | Ratio of component activity in the last year of that window to the first, minus one |
| `SIM_ACCESS` | Absolute difference in cumulative activity between the two components |
| `SIM_TECH` | Weighted dissimilarity of the two IPC codes at section, class and subclass level, with weights 0.5, 0.3 and 0.2 |
| `INVENT_DIVER` | Shannon entropy of the IPC section distribution across the early inventions |
| `INVENT_APPL` | Mean number of IPC codes per early invention |
| `ATTENT_SIZE` | Count of early inventions |

`SIM_TECH` rises as the two components become more distant, since each term indicates a
difference at that level of the hierarchy rather than a match.

## References

Chen, W., Ma, Y., Ba, Z., and Li, G. (2025). Predicting reuse patterns of novel technologies:
the impact of technology components and early inventions on technology trajectories.
*Scientometrics*. https://doi.org/10.1007/s11192-025-05449-1

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and
Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical
machine translation. *Proceedings of EMNLP*.

Wu, X., and Cheng, Q. (2021). Fractal autoencoders for feature selection. *Proceedings of the
AAAI Conference on Artificial Intelligence*.

## Data source

Patent records come from the United States Patent and Trademark Office Open Data Portal and are
in the public domain. The derived technology corpus is constructed by the code in this
repository.
