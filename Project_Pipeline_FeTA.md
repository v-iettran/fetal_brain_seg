# **FeTA Fetal Brain Tissue Segmentation — Project Pipeline**

*Written for a mixed-background team. Terms in **bold italics** are defined in the
Glossary at the end. If something is unclear, that's a bug in this document — say so.*

---

## **1. What we are doing**

We have MRI scans of 80 unborn babies' brains. For each scan, a human expert has
already coloured in which parts are which tissue — grey matter, white matter, and so
on. Our job is to build software that does that colouring automatically.

This is called ***segmentation***. Unlike a classification task (where a model outputs
one answer per image, e.g. "tumour / no tumour"), here the model outputs **one answer
per 3D pixel**. Each scan is 256 × 256 × 256 = ~16.8 million ***voxels***, and the model
assigns each one to a tissue class.

### **The eight classes**

| Label value | Tissue |
| ----- | ----- |
| 0 | Background (outside the brain) |
| 1 | External cerebrospinal fluid (eCSF) |
| 2 | Grey matter (GM) — the cortex, the outer folded layer |
| 3 | White matter (WM) — the connective tissue underneath |
| 4 | Ventricles — fluid-filled cavities inside the brain |
| 5 | Cerebellum — the small structure at the back |
| 6 | Deep grey matter — clusters deep inside (thalamus, basal ganglia) |
| 7 | Brainstem / spinal cord |

### **The two tracks**

**Track A — Reproduction.** Rebuild the method that placed 3rd in the FeTA 2021
challenge (team `pengyy`) and establish it as our reference. This is the main body of
the project and involves everyone.

**Track B — Discovery.** A separate experiment run by Viet and James using
**OpenEvolve**, an automated tool that proposes and tests code changes. Described
briefly in §8 and in full in a separate document. It shares our data splits and our
evaluation code, but otherwise runs independently.

---

## **2. What data we actually have — read this before planning anything**

The folder `mri_gz/` contains **160 files: 80 subjects × 2 files each.**

| File | What it is |
| ----- | ----- |
| `sub-001_rec-mial_T2w.nii.gz` | The MRI scan (greyscale intensities) |
| `sub-001_rec-mial_dseg.nii.gz` | The expert's tissue labels (integers 0–7) |

`.nii.gz` is the **NIfTI** format — the standard container for 3D medical images. Load
it in Python with `nibabel`.

### **What we know about each subject**

Everything comes from the **filename** and the **file header**. There is no spreadsheet.

* **Subject ID** — `sub-001` … `sub-080`
* ***Reconstruction method*** — `rec-mial` for subjects 001–040, `rec-irtk` for 041–080.
  The raw MRI of a moving fetus is unusable, so a software pipeline stitches many blurry
  2D slices into one sharp 3D volume. Two different pipelines were used, and they produce
  visibly different-looking images. **This is the single most important variable in the
  dataset** — in the original challenge, the ranking of competing methods changed
  completely depending on which pipeline's images were tested.
* **Image dimensions** — 256 × 256 × 256 for all subjects
* ***Voxel spacing*** — how many millimetres one voxel represents. **This varies between
  subjects, from 0.43 mm to 0.70 mm.** It is stored in the header, not the filename.
* **Data type** — some scans are stored as `int16`, others as `float32`. All label maps
  are `float32`.

```python
import nibabel as nib
img = nib.load("mri_gz/sub-001_rec-mial_dseg.nii.gz")
print(img.header)          # dimensions, spacing, data type
print(img.header.get_zooms())   # voxel spacing in mm, e.g. (0.547, 0.547, 0.547)
```

### **What we do NOT have**

> The original FeTA release ships a file called `participants.tsv` containing
> **gestational age** (how many weeks pregnant) and a **pathology flag** (whether the
> brain is developing normally or has an abnormality). **This copy does not include it.**

This matters, because both are scientifically important:

* **Gestational age** drives everything. A 21-week brain and a 35-week brain look
  radically different — the cortex is nearly smooth at 21 weeks and heavily folded by 35.
  The contrast between grey and white matter also inverts over this period. Segmentation
  difficulty is therefore strongly age-dependent.
* **Pathology** matters because abnormal brains are much harder to segment, and roughly
  half this dataset is pathological.

### **How we handle the gap**

**First: try to get the file.** The FeTA dataset is distributed via Synapse
(`syn25649159`) and described in Payette et al., *Scientific Data* 8:167 (2021).
Ask the module supervisor whether the full release can be obtained. **Ilaria — this is
your first task, and do it in week 1**, because everything below is a fallback.

**If we cannot get it, we use a measurable proxy:**

| Missing variable | Proxy we compute ourselves | Status |
| ----- | ----- | ----- |
| Gestational age | ***Intracranial volume (ICV)*** = total volume of all non-background labels | **Good.** Brain volume grows monotonically and steeply with gestational age. Ranking subjects by ICV approximately ranks them by maturity. |
| Cortical maturity | ***Surface-to-volume ratio*** of the grey matter label | **Reasonable.** Folding increases surface area faster than volume, so this rises with age. |
| Pathology | Ventricle volume ÷ ICV (enlarged ventricles = ventriculomegaly, the most common abnormality here) | **Exploratory only.** Flags one pathology, not all. Never present this as a pathology label. |

⚠️ **Be honest about this in the report.** Write: *"Gestational age and pathology labels
were unavailable in our copy of the dataset. We use intracranial volume derived from the
reference segmentation as a proxy for maturational stage, and report all
age-related analyses on that basis."* A stated limitation is fine. A hidden one is not.

---

## **3. How we split the data**

We divide the 80 subjects into three groups that are **never mixed**:

| Split | Size | Used for |
| ----- | ----- | ----- |
| **Training** | 40 | The model learns from these |
| **Tuning** | 20 | We try different settings and pick the best on these |
| **Test** | 20 | **Sealed.** Opened once, at the very end, for the final numbers |

### **Why three and not two**

If you try 30 different settings and report the best score on the same data you used to
choose, that score is optimistically biased — you have partly measured luck. The tuning
split absorbs that bias; the test split gives an honest number.

**The test split is sealed.** Nobody looks at it, plots it, or computes a metric on it
until the final models are frozen. This is not bureaucracy — it is the difference
between a result and a number.

### **What we balance the split on**

**Reconstruction method — mandatory.** Exactly 20 mial + 20 irtk in training, 10 + 10 in
tuning, 10 + 10 in test. Since mial is subjects 001–040 and irtk is 041–080, this is easy
to get right and disastrous to get wrong. An unbalanced split would mean training mostly
on one image style and testing on another, and we would not be able to tell whether a
result was about the model or about the split.

**ICV tertile — recommended.** Split subjects into thirds by intracranial volume
(small / medium / large ≈ younger / middle / older) and balance those across the three
splits too. This stops all the youngest brains landing in the test set.

The split is generated **once**, written to `results/splits.json`, and never regenerated.
Both tracks import that file.

---

## **4. Repository structure**

No notebooks. Every piece of work is a Python script with a command-line interface that
reads inputs from disk and writes results to disk as JSON.

```
feta/
│
├── config.py                # Seed (=42), data paths, class names. Nothing else hardcodes a path.
│
├── src/
│   ├── explore.py           # Ilaria — inspect data, compute proxies, QC
│   ├── splits.py            # Caolan — generate splits.json
│   ├── convert.py           # Caolan — put data into nnU-Net's expected format
│   ├── train.py             # Caolan — training
│   ├── tuning.py            # Caolan — Optuna hyperparameter search
│   ├── predict.py           # Caolan — run a trained model on new scans
│   ├── metrics.py           # Sonia — Dice, HD95, etc. Pure functions.
│   ├── validate.py          # Sonia — evaluation harness + statistics
│   ├── interpret.py         # Albee — error maps, volume analysis, figures
│   └── report_tables.py     # Noma — turns results/*.json into report tables
│
├── openevolve/              # Viet & James — separate spec
└── results/                 # All outputs. Committed to git (they're small JSON files).
```

**Why no notebooks:** results must be reproducible by re-running a command. Notebooks
hold hidden state — a cell run out of order silently changes your numbers, and nobody can
tell afterwards. Scripts also let five people work without merge conflicts.

**How the pieces connect:** Sonia's `metrics.py` is imported by Caolan's tuning loop,
by her own `validate.py`, and by the OpenEvolve evaluator. One implementation used
everywhere means the numbers are always comparable.

---

## **5. Who does what**

### **5.1 Ilaria — Data exploration & quality control**
**File:** `src/explore.py`

> **Scope note.** You are *not* writing a preprocessing pipeline. nnU-Net inspects the
> dataset itself and decides its own resampling and normalisation. Anything you build
> would be overwritten. Your job is to **understand and describe** the data, and to
> produce the metadata everyone else depends on.

**Step by step:**

1. **Try to obtain `participants.tsv`** from the supervisors or the Synapse release.
   Do this first; report the outcome to the team by end of week 1.
2. **Read every header.** Loop over all 160 files, record: dimensions, voxel spacing,
   data type, and the affine matrix. Save to `results/fingerprint.json`.
3. **Check the label maps are valid.** They're stored as `float32`, so round to integer
   before use. Confirm the only values present are 0–7. Confirm every subject has all
   7 tissue classes present (if any are missing, that's a finding — flag it).
4. **Compute the proxy variables** for every subject: intracranial volume, per-class
   volumes, ventricle fraction, GM surface-to-volume ratio. Remember to multiply voxel
   counts by voxel volume — spacing differs between subjects, so raw voxel counts are
   **not** comparable across subjects. Save to `results/cases.csv`.
5. **Quantify class imbalance.** Report each tissue as a percentage of intracranial
   volume. Expect brainstem and cerebellum to be tiny (~1–2%). This is why the model
   needs special sampling — a randomly placed training patch usually contains no
   brainstem at all.
6. **Visual QC.** View a mid-brain slice of every subject. Flag blurry or artefact-heavy
   reconstructions. Note whether mial and irtk images look systematically different.
7. **Label quality audit** *(this becomes a section of the paper).* The original
   annotators drew labels on only every 2nd–3rd slice, mostly in the **axial** plane
   (cerebellum and brainstem in **sagittal**), and software filled the gaps. Different
   people annotated different tissue classes. Look at the label maps from the side
   (coronal and sagittal views) and you should see staircase artefacts. Capture examples.
   This is important: the challenge organisers reported that some methods produced
   visually excellent output but scored mid-range *because the ground truth itself was
   imperfect*.

**Deliverables:** `fingerprint.json`, `cases.csv`, figures (ICV distribution, per-class
volume fractions, example slices across the ICV range, label artefact examples).

---

### **5.2 Caolan — Splitting, training, tuning**
**Files:** `src/splits.py`, `src/convert.py`, `src/train.py`, `src/tuning.py`

**Step by step:**

1. **Generate the split** (needs Ilaria's `cases.csv` for the ICV tertiles). Balance on
   reconstruction method and ICV tertile. Write `results/splits.json`. **Do this once.**
2. **Convert to nnU-Net format.** nnU-Net expects a specific folder layout and a
   `dataset.json` describing the classes. Since our copy has no
   `dataset_description.json`, you write this yourself: 8 classes, one input channel,
   file suffix `.nii.gz`. Only the 40 training subjects go into the nnU-Net raw folder;
   tuning and test subjects are held elsewhere.
3. **Run planning and preprocessing** (`nnUNetv2_plan_and_preprocess`). nnU-Net analyses
   the data and decides patch size, resampling target, and network depth automatically.
   Record what it chose — this is a result worth reporting.
4. **Train three configurations**, in this order:

   | Config | Purpose |
   | ----- | ----- |
   | Vanilla 3D U-Net (Çiçek 2016) | Floor — what a basic model achieves |
   | Stock nnU-Net 3d_fullres | The modern standard |
   | `pengyy` config: as above but **48 base features** instead of 32 | The 3rd-place challenge method |

   nnU-Net runs its own internal **5-fold cross-validation** on the 40 training subjects.
   A ***fold*** means: split the 40 into 5 groups, train 5 models each holding one group
   out. The 5 models are then averaged at prediction time — this averaging *is* the
   "ensemble learning" listed in the challenge paper.

   > **On the 48 features:** the challenge paper reports 72,142,688 parameters for
   > `pengyy`, while two other teams report 31,199,584 for stock nnU-Net on the same
   > data. The ratio implies the base feature count was raised from 32 to 48. Build the
   > network and check the parameter count — if you land near 72.1 M, we've inferred
   > correctly. If not, say so in the report.

5. **Establish the noise floor before tuning.** Train the same configuration three times
   with different random seeds. The spread of the results is your measurement noise.
   **Any later "improvement" smaller than about twice that spread is not real.** This
   takes three training runs and saves the whole team from over-claiming.
6. **Hyperparameter tuning with Optuna** on the 20 tuning subjects. A
   ***hyperparameter*** is a setting you choose rather than learn — learning rate,
   optimiser, etc.

   | Parameter | Range |
   | ----- | ----- |
   | Learning rate | 1e-3 to 1e-1 (log scale) |
   | Weight decay | 1e-6 to 1e-3 (log scale) |
   | Optimiser | SGD+Nesterov / Adam / AdamW |
   | Dice-to-cross-entropy weight ratio | 0.25 to 4.0 |
   | Foreground oversampling fraction | 0.2 to 0.6 |

   Run trials at a **shortened schedule** (250 epochs instead of 1000) with Optuna's
   `MedianPruner` to kill bad trials early. Budget ~25–30 trials.

   ⚠️ **Report the retrained model, not the best trial.** Take the winning settings,
   retrain from scratch at the full 1000 epochs, and report that. Reporting the maximum
   over 30 trials inflates the score by roughly one standard deviation of the noise.

7. **Predict on tuning and test splits** using `predict.py`, saving label maps for Sonia
   and Albee to analyse.

**Deliverables:** `splits.json`, model checkpoints, `results/trackA/*.json`, Optuna study
and trial-history figure.

---

### **5.3 Sonia — Metrics and validation**
**Files:** `src/metrics.py`, `src/validate.py`

> You are writing **modules other people import**, not a standalone analysis. Your API
> is a contract — agree it with Caolan and Viet in week 1 and then keep it stable.

**Step by step:**

1. **Write `metrics.py` as pure functions**: `(prediction_array, ground_truth_array,
   voxel_spacing) → dict of scores`. No file loading, no plotting, no global state.

   | Metric | What it measures |
   | ----- | ----- |
   | **Dice** | Overlap: `2 × (voxels in both) / (voxels in A + voxels in B)`. 0 = no overlap, 1 = perfect. Our primary metric. |
   | **HD95** | Boundary error: the 95th percentile of the distance from each predicted surface point to the true surface. In **millimetres**. Catches a correct-looking region that's slightly misplaced. |
   | **Volume Similarity** | Whether the total volume is right, ignoring whether it's in the right place. Clinically relevant, because volume is what doctors measure. |
   | **Euler characteristic difference** | Topological error — does the predicted structure have the right number of holes and connected pieces? A segmentation with the right shape but spurious holes scores badly here and well on Dice. |

2. **Fix and document the conventions** (once, in a docstring):
   * Compute per class, then average over classes 1–7. **Exclude background** — it's
     ~85% of the volume and would swamp everything.
   * If a class exists in the ground truth but the model predicted zero voxels of it:
     Dice = 0, VS = 0, HD95 = a defined penalty value. Record which value.
   * ⚠️ **HD95 must use the real voxel spacing from the header**, which varies per
     subject. Using voxel counts instead of millimetres makes subjects incomparable.

3. **Unit-test before any model exists.** Generate two synthetic spheres with known
   overlap — you can compute the true Dice by hand — and assert your function returns it.
   Test the empty-prediction and identical-input edge cases too. This unblocks you from
   waiting on Caolan.

4. **Write `validate.py`** — takes a folder of predictions and produces:
   * Per-subject × per-class scores as JSON
   * **Subgroup analysis:** scores broken down by **reconstruction method** (the real
     comparison) and by **ICV tertile** (the age proxy). This is where the interesting
     variation lives.
   * **Statistical comparison** between configurations: Wilcoxon signed-rank test paired
     by subject, with Holm correction across the 7 classes. Also bootstrap the ranking
     to show whether it's stable — the challenge organisers did exactly this.
   * **Failure detection:** flag every case where a class was predicted with zero voxels.
     Expect this on brainstem and cerebellum.

**Deliverables:** tested modules with a documented API, per-class boxplots, subgroup
tables, significance maps.

---

### **5.4 Albee — Interpretability and clinical analysis**
**File:** `src/interpret.py`

> Interpretability for segmentation means **showing where and how the model is wrong**,
> and asking whether the errors are clinically acceptable. Grad-CAM and saliency maps
> don't apply — the model already tells you exactly which voxels it chose.

**Step by step:**

1. **Error maps.** For the worst-performing cases, overlay false positives (predicted
   tissue that isn't there) in one colour and false negatives (missed tissue) in another,
   on top of the greyscale scan. Show axial, coronal and sagittal views.
2. **Boundary error visualisation.** Render the distance-to-truth on the predicted
   surface as a heat map. This shows *where* the HD95 number comes from, which a single
   number cannot.
3. **Confusion analysis.** Build a 8×8 matrix of which tissue gets mistaken for which.
   Expected pattern: grey matter ↔ white matter (their MRI contrast changes and weakens
   as the brain matures), and deep grey matter ↔ white matter at the poorly-defined
   lateral borders.
4. **Intracranial volume agreement.** Sum all non-background predicted labels and compare
   to the truth with a **Bland–Altman plot** (difference vs mean). Total brain volume is
   the measurement clinicians actually use, so this is the most clinically meaningful
   figure in the report.
5. **Volume trajectories.** Plot each tissue's volume against ICV (our maturity proxy),
   predicted vs ground truth. If the curves diverge at the extremes, the model is
   systematically biased for the youngest or oldest brains.
6. **Failure gallery.** The 5 worst cases by mean Dice, annotated with reconstruction
   method and ICV tertile. Look for a pattern: are failures concentrated in one
   reconstruction pipeline, or in the smallest brains?
7. **"Better than ground truth" cases** *(the highest-value figure in the project).*
   Look for cases where the prediction is anatomically more plausible than the expert
   annotation — smooth boundaries where the ground truth has interpolation staircases,
   for instance. The challenge organisers documented this happening. It reframes our
   whole results section: at this point the ceiling is annotation quality, not model
   capacity.
8. **Clinical review.** Take the failure gallery and the "better than ground truth"
   cases to a clinician or the supervisors for qualitative comment. Costs an hour and
   substantially strengthens the discussion.

**Deliverables:** figures for the results and discussion sections, plus a short written
assessment of whether errors are clinically consequential.

---

### **5.5 Noma — Paper**
**File:** `src/report_tables.py`

Structure:

1. **Introduction** — why fetal MRI, why automatic segmentation, what FeTA is
2. **Data** — dataset description, the missing-metadata limitation and our proxies,
   Ilaria's label-quality audit
3. **Methods** — Track A configurations; Track B in brief
4. **Experiments** — splits, metrics, tuning protocol, noise floor
5. **Results** — the three configurations, tuning gains, subgroup breakdown
6. **Interpretability** — error analysis, volumetry, clinical review
7. **Discussion** — the field has plateaued near inter-rater agreement; annotation
   quality is the real ceiling
8. **Limitations & Future Work**

`report_tables.py` generates every table directly from `results/*.json`, so a number in
the paper can never drift from the number on disk. Nobody types results by hand.

---

## **6. Reference points**

| Reference | Score | Note |
| ----- | ----- | ----- |
| `pengyy`, FeTA 2021 | Dice 0.774 ± 0.182 | On the **hidden** challenge test set — **not comparable to ours** |
| Stock nnU-Net | our measurement | Modern standard |
| Vanilla 3D U-Net | our measurement | Floor |

> ⚠️ **Say this explicitly in the paper.** The FeTA 2021 test set was never released. Our
> 20 test subjects come from the same public pool we trained on, so our Dice will be
> *higher* than 0.774. Comparing the two numbers directly is invalid, and a reader will
> notice if we don't say so first.

---

## **7. Order of work**

```
Ilaria: obtain participants.tsv?  →  cases.csv + fingerprint.json
                                            ↓
Caolan: splits.json  ←──────────────────────┘
        ↓
Sonia: metrics.py (can start immediately, using synthetic test data)
        ↓
Caolan: convert → plan → train 3 configs → noise floor → Optuna → predict
        ↓
Sonia: validate.py → per-class, subgroup, significance
        ↓
Albee: error maps, volumetry, failure gallery, clinical review
        ↓
                  SEALED TEST SET — opened once
                             ↓
                       Noma: paper
```

**Critical path:** `splits.json` and `metrics.py` block nearly everything downstream.
Both should exist by end of week 1. Sonia is not blocked by anyone — synthetic test data
lets her build and verify the metrics before a single model is trained.

---

## **8. Track B — OpenEvolve (Viet, James)**

*Brief summary; full detail in a separate specification.*

**The question:** nnU-Net is the result of years of careful human refinement of the
original 2016 U-Net. Can an automated system rediscover those refinements on its own?

We give **OpenEvolve** — a tool that uses a language model to propose code changes,
trains each proposed version, and keeps the ones that score well — the plain 2016 U-Net
as a starting point. Before running anything, we write down eight specific design
decisions that separate the 2016 model from nnU-Net (a normalisation change, an
activation change, deep supervision, a different loss, a different optimiser, and so on).
We then measure **which of the eight are rediscovered, in what order, and at what
computational cost.**

Two control experiments guard the claim: asking the language model to improve the network
with no evolution at all (does it simply already know the answers from having read the
papers?), and running the same mutations with random rather than score-based selection
(is the scoring doing any work?).

Track B imports `results/splits.json` and `src/metrics.py` and uses the same sealed test
set, so its numbers sit in the same tables as Track A's. It runs on a dedicated GPU and
does not compete for Track A's compute.

---

## **9. Limitations to state in the report**

Naming these pre-empts the obvious criticisms. They are deliberate choices, not oversights.

1. **No gestational age or pathology labels** in our copy of the dataset. We substitute
   intracranial volume as a maturity proxy and cannot analyse pathological versus
   neurotypical performance separately.
2. **Our test scores are not comparable to the FeTA leaderboard** — the challenge test
   set is hidden and ours is in-distribution.
3. **5-fold on 40 subjects, not 10-fold on 80.** `pengyy` trained on all 80 public
   subjects; we hold out 40 for tuning and testing. Our ensemble is smaller and our
   training set is half the size, so we expect to score below their setup on equivalent
   data.
4. **The 48-feature setting is inferred** from the reported parameter count, not
   documented. We verify by construction and report any discrepancy.
5. **Our HD95 is not comparable to the FeTA 2021 paper's**, whose implementation was
   buggy and reported near-maximum Hausdorff distance instead of the 95th percentile.
6. **Label noise is the real ceiling.** The organisers state that accuracy is approaching
   inter-annotator disagreement. Small differences between our configurations are
   probably not meaningful — which is why we report a noise floor and significance tests
   rather than raw rankings.
7. **Single centre.** All FeTA 2021 data comes from one hospital in Zurich, so we cannot
   assess generalisation to other scanners. The reconstruction-method comparison is our
   closest available proxy.

---

## **10. Glossary**

| Term | Meaning |
| ----- | ----- |
| **Voxel** | A 3D pixel. Our images are 256³ ≈ 16.8 million voxels. |
| **Voxel spacing** | Physical size of one voxel in mm. Varies per subject here (0.43–0.70 mm). |
| **Segmentation** | Assigning a class label to every voxel. |
| **NIfTI (`.nii.gz`)** | Standard 3D medical image file format. Read with `nibabel`. |
| **Reconstruction method** | Software (`mial` or `irtk`) that built a sharp 3D volume from blurry 2D slices of a moving fetus. |
| **Dice coefficient** | Overlap score, 0 to 1. Our primary metric. |
| **HD95** | 95th-percentile boundary distance error, in mm. |
| **Intracranial volume (ICV)** | Total volume of all brain tissue. Our proxy for gestational age. |
| **U-Net** | The standard segmentation network: compress the image down, expand it back up, with shortcut connections preserving fine detail. |
| **nnU-Net** | A U-Net plus an automatic system that configures itself from the dataset. Current standard for medical segmentation. |
| **Patch** | A small cube (128³) cut from the full volume. The full 256³ volume doesn't fit in GPU memory. |
| **Epoch** | One training cycle. We train for 1000. |
| **Fold / cross-validation** | Splitting training data into k parts and training k models, each holding one part out. The k models are averaged at prediction time. |
| **Ensemble** | Averaging several models' predictions. Almost always better than any single model. |
| **Hyperparameter** | A setting you choose rather than learn (learning rate, optimiser, ...). |
| **Optuna** | A library that searches hyperparameter combinations intelligently instead of by grid search. |
| **Loss function** | The number the model minimises during training. Ours combines Dice and cross-entropy. |
| **Deep supervision** | Attaching extra loss terms at intermediate network depths so gradients reach early layers. |
| **Foreground oversampling** | Deliberately choosing training patches that contain brain rather than background. Without it, most patches would be empty. |
| **Noise floor** | The run-to-run variation from random seeds alone. Improvements smaller than this are not real. |
| **OpenEvolve** | Tool that uses a language model to propose code changes, scores them, and keeps the best. Track B only. |
