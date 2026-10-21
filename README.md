# Boosting Algorithms

Bagging builds its trees side by side and aggregates them. **Boosting** builds them one after
another, and every tree is told where the ones before it went wrong. This repository works
through four of them — AdaBoost, XGBoost, CatBoost and LightGBM — one notebook each, and
each one standing on its own.

It runs on two datasets. **45,222 census records**, where the target is whether a person
earned more than $50,000, carries AdaBoost and CatBoost; its eight text columns are what
notebook 03 needs. **20,640 California districts**, where the target is the median house
value, carries XGBoost and LightGBM, and the comparison that closes notebook 03: it is large
enough that early stopping pays for its validation slice, and that a 500-tree race can be run
twice at different row counts. Both ship in `data/`, so every notebook runs on its own and
offline.

Every algorithm here does both jobs. The split between classification and regression is a
teaching choice, not a property of the libraries, and notebook 03 closes by running every
ensemble taught so far on the same regression problem.

```
01  adaboost: weights on rows, a vote of stumps      (census, classification)
        |
        v
02  xgboost: fit the residual, then pay for complexity  (california, regression)
        |
        v
03  catboost: categorical columns, why target encoding leaks,
    and every ensemble so far compared       (census, then california)
        |
        v
04  lightgbm: lopsided trees, binned features   (california, OPTIONAL)
```

## Learning Objectives

By the end of this repository, you should be able to:

- Implement the AdaBoost weight update, and explain why the weak learner that has just run
  scores exactly 0.5 on the weights it leaves behind.
- Choose `learning_rate` and `n_estimators` together, and use early stopping to set the
  second of them in a single fit.
- Diagnose overfitting from the gap between training and test error, and close it with the
  complexity penalties of XGBoost and LightGBM.
- Identify a target-encoded column that leaks, and apply CatBoost's `cat_features` instead.
- Tune `num_leaves` by cross-validation, and decide whether a difference measured on a test
  set is large enough to believe.
- Compare the four algorithms on one regression problem, and decide between them on
  cardinality, table size and fitting time rather than on a single score.

## Learning Path

**Scope** marks the core path every student is expected to complete. Optional lessons stay in
the repository and are worth returning to, but the day does not depend on them. **Units** are
indicative pacing, where one unit is 45 minutes.

| File / Folder | Description | Scope | Units |
| --- | --- | --- | --- |
| [**1 - Adaptive Boosting (AdaBoost)**](01_adaboost.ipynb) | A stump that predicts `<=50K` for everyone, boosted to 0.8485 accuracy. The weight update written out by hand, then the same result from `AdaBoostClassifier`, then the finding that 150 stumps still lose to one pruned tree and depth 4 is what wins. | Core | 1.75 |
| [**2 - Extreme Gradient Boosting (XGBoost)**](02_xgboost.ipynb) | 0.458 against a random forest's 0.472, untuned. The learning rate and the tree count as one setting, early stopping that finds 992 trees in a single fit, and the complexity penalties that make the training error get worse and the test error better — two of the four charging so much they finish behind the defaults. | Core | 2.25 |
| [**3 - Categorical Boosting (CatBoost)**](03_catboost.ipynb) | A column of random payroll ids, target-encoded the obvious way, costs 0.151 of test AUC. Ordered target statistics, ordered boosting, symmetric trees, and `cat_features` with no encoding step at all. Closes by scoring every ensemble taught so far on one regression table. | Core | 2.25 |
| [**4 - Light Gradient Boosting Machine (LightGBM)**](04_lightgbm.ipynb) | Leaf-wise growth, so `num_leaves` replaces `max_depth` and has to be matched to the table. Histogram binning and what coarsening it costs, and a 500-tree race run at two row counts to show what the binning buys. | Optional | 1.75 |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Data**](data/) | `census.csv`: 45,222 records, 8 categorical and 5 numeric columns, with `income` as the target. `concrete.csv`: 1,030 mixes, seven ingredients and an age in days, with `compressive_strength` in MPa as the target. Both are modified from the UCI originals; see the references below for what changed. |
| [**Solutions**](solutions/) | Reference solutions for every notebook. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd amle-boosting-algorithms`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebook

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open `01_adaboost.ipynb`, selecting the Python environment created by `uv sync` as the kernel. Work through the notebooks in order. All four are the core path.

> [!CAUTION]
> macOS only: `uv sync` installs XGBoost and LightGBM, but neither bundles the system OpenMP runtime it links against. If a notebook reports that `libxgboost.dylib` or `lib_lightgbm.dylib` could not be loaded, install it once with `brew install libomp`. The Linux and Windows wheels ship OpenMP, so no extra step is needed there. See the [XGBoost](https://xgboost.readthedocs.io/en/stable/install.html) and [LightGBM](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html) installation notes.

## References & Further Reading

- [**scikit-learn: Ensembles**](https://scikit-learn.org/stable/modules/ensemble.html): Boosting alongside bagging and voting, with the parameters that matter and the reasoning behind them. Note that version 1.9 renamed `AdaBoostClassifier`'s weak learner argument to `estimator` and dropped `algorithm=`, which most tutorials have not caught up with.
- [**Freund and Schapire, A Decision-Theoretic Generalization of On-Line Learning (1997)**](https://doi.org/10.1006/jcss.1997.1504): AdaBoost, including the weight update notebook 01 implements.
- [**Friedman, Greedy Function Approximation: A Gradient Boosting Machine (2001)**](https://doi.org/10.1214/aos/1013203451): Where fitting the residual comes from, and the learning rate with it.
- [**Chen and Guestrin, XGBoost: A Scalable Tree Boosting System (2016)**](https://doi.org/10.1145/2939672.2939785): The regularized objective of notebook 02, section 6.
- [**Prokhorenkova et al., CatBoost: Unbiased Boosting with Categorical Features (2018)**](https://papers.nips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html): Ordered target statistics and ordered boosting, which notebook 03 is built around.
- [**Ke et al., LightGBM: A Highly Efficient Gradient Boosting Decision Tree (2017)**](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html): GOSS, EFB, and the 20x speed claim notebook 04 tests.
- [**LightGBM: Parameters Tuning**](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html): The official guidance on `num_leaves`, `min_data_in_leaf` and `max_bin`, and what to do when a model overfits.
- [**AdaBoost implementation by Jaime Pastor**](https://github.com/jaimeps/adaboost-implementation): The from-scratch loop notebook 01 is adapted from, rewritten here in seeded NumPy with a clipped weighted error.
- [**UCI: Adult**](https://archive.ics.uci.edu/dataset/2/adult): Extracted from the 1994 US census by Barry Becker and Ronny Kohavi, licensed CC BY 4.0. **Modified here:** the 48,842 original rows are reduced to the 45,222 with no missing values, two columns are renamed, and the leading spaces inside its category values have been stripped.
- [**UCI: Concrete Compressive Strength**](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength): Collected by I-Cheng Yeh, licensed CC BY 4.0. **Modified here:** converted from the original `.xls` to CSV and the nine columns renamed to short snake_case names. No rows or values were changed.
