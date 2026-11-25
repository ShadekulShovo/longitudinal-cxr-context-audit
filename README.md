# Longitudinal chest-radiograph context audit

This repository provides cleaned numerical analysis code for assessing whether
patient history improves ranking while changing onset and resolution decisions.
It contains no images, labels, reports, identifiers, fitted heads, features,
predictions or other study-level data. The numerical extract contains aggregates
only. No clinical deployment or patient-level explanation benefit is claimed.

The code is a cleaned copy of the analysis code, with no claim of byte identity
to any execution record. Tests and documentation cover the cleaned tree.

## Repository

https://github.com/ShadekulShovo/longitudinal-cxr-context-audit

```sh
git clone https://github.com/ShadekulShovo/longitudinal-cxr-context-audit
cd longitudinal-cxr-context-audit
python -m pip install -r requirements.txt
```

The folder is supplied without repository history. Publication at this address
is the authors' responsibility; availability has not been asserted or checked.

## Requirements and tests

The pinned versions in requirements.txt describe the local verification
environment (Python 3.12). They are not a claim that all historical feature
extractions used identical versions. External feature processing used an A100
cloud runtime; local extraction used an RTX 3060 with 12 GB memory. Metrics and
figure generation run on CPU. Bootstrap seed: 20261115. Matched-replacement seed:
20261116. Logistic fitting seed: 11.

The cleaned synthetic suite passed **23 tests**. It checks exact pair-count
identities, tied scores, whole-patient resampling, undefined inputs, strict
history, donor constraints, original float32 history coordinates, fixed lexical
rules, exact logit reconstruction and decision-cell algebra.

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

On Windows PowerShell set `$env:PYTHONPATH="src"` before the command.

## Dataset access

Obtain each dataset directly from its provider and follow its current terms:

* TAIX-Ray: https://huggingface.co/datasets/TLAIM/TAIX-Ray
* NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
* PadChest: https://bimcv.cipf.es/bimcv-projects/padchest/
* ReXGradient-160K: https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K
* BioViL-T: https://huggingface.co/microsoft/BiomedVLP-BioViL-T

PadChest requires the provider's access process. ReXGradient requires an account
and acceptance of its non-commercial access agreement. No authorship or
institutional ethics determination is implied by these access instructions.
ReXGradient-160K was provided to Harvard by Gradient Health, recognized here
with the data repository as the source of those data.

## Reproduce the tables and figures

```sh
python scripts/make_tables.py --metrics results/metrics.json --output tables
python scripts/make_figures.py --metrics results/metrics.json --output figures
```

Both commands use only the numerical extract. The synthetic worked example is
provided with the manuscript as a separate illustrative asset, not real data.
The full numerical extract retains confidence intervals, valid replicate counts
and reliability flags. A missing interval is not zero uncertainty. Different
cohorts and reference models are not pooled.

`scripts/extract_metrics.py` accepts aggregate analysis summaries through four
explicit input arguments and emits the numerical extract. Those source
summaries are intentionally not distributed because their administrative
records are outside the numerical release. Rebuilding the published tables and
plots does not require them.

## Rerun analyses with legitimate access

Image acquisition and encoder execution are not bundled. Use the providers'
released encoders and their specified preprocessing, and keep all study-level
arrays in your own permitted storage. The manuscript specifies label mappings,
history ordering and split rules. Zero patient overlap must be asserted before
fitting, with strictly earlier history; NIH uses follow-up order rather than
dates. Do not reuse test outcomes to choose a model or threshold.

`src/graded_history.py` supplies the original eight-finding graded-history
calculation. `src/models.py` supplies the five-finding binary-history calculation,
logistic selection, matched donors and aggregate statistics. `src/metrics.py`
supplies generic pair decomposition, thresholds, calibration and paired
bootstrap. `src/linear_inference.py` retains transformed-coordinate inference.
`src/linear_audit.py` supplies the exact saved-logit reconstruction gate and
override/rescue audit. `src/report_language.py` supplies fixed literal report
groups and paired report-group contrasts.

For an external five-finding audit, prepare a local NPZ with aligned `y`, `prior`,
`patients`, `scores_M1`, `scores_M4`, `single_M1`, `single_M4`,
`stratified_M1`, `stratified_M4`. Labels and prior states are binary arrays of
shape (studies, five findings), scores are saved logits, single cuts have five
elements, and stratified cuts have shape (two prior states, five findings).
Optional `history_swap` and `image_swap` are saved replacement logits; optional
`comparative` is a fixed report-group flag. Use the same patient ordering for
all arrays. Current and prior image replacements must be re-encoded jointly for
BioViL-T; replacing only its standalone feature block is not equivalent.

```sh
python scripts/run_audit.py --input private_arrays.npz --output private_results
```

The run is resumable using `--resume`, writes progress and a completion receipt,
and refuses to overwrite completed results. Never upload its private inputs or
row-level intermediate files with this repository. For linear audits, supply
saved transformed coordinates, coefficients, intercepts, reference vectors and
original saved logits to `linear_decomposition`; require original output
reproduction before attribution. The manuscript reports NIH attribution as
stopped and TAIX attribution as unavailable, not as successful results.

## Citation and licence

Repository: https://github.com/ShadekulShovo/longitudinal-cxr-context-audit

Manuscript: When patient history helps a chest radiograph model, and when it
hurts: a four-dataset audit of ranking, operating points and reliance.
