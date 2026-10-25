# Supervised Machine Learning Lab Journal

**Author:** Harshit Jaiswal (Roll No. A049)  
**Course instructor:** Pratik A. Desai Sir  
**Deliverable:** `A049_Harshit_Jaiswal_sml_lab_journal.ipynb`

A single Jupyter notebook that consolidates all ten laboratory practicals taught in Semester 1 into one sequential, reproducible journal. Every code cell is preceded by a short explanation (what it does, why it is used, how it works), and every lab ends with a small experiment of the author's own design.

---

## Table of Contents

1. Overview
2. Tech Stack and Setup
3. Experiments and Evaluation Methodology
4. Results and Analysis
5. Failure Analysis and Limitations
6. Testing, Roadmap and References
7. Acknowledgement

---

## 1. Overview

### What it is

This repository contains a complete, executable record of ten supervised-learning laboratories, ordered from the simplest model (k-Nearest Neighbours) to a feed-forward neural network built with Keras. The notebook runs top to bottom and reproduces every figure, table and score discussed below.

### The problem it addresses

Lab work is normally scattered across separate files with inconsistent structure, which makes it difficult to review, re-run or compare. This journal solves that by:

- placing all ten labs in one notebook, in the order they were taught;
- explaining every code cell in plain language so the reasoning, not only the output, is documented;
- applying a uniform workflow to each lab (load data, explore, split, train, evaluate, interpret);
- adding one original experiment per lab so the work goes beyond the supplied exercises.

### Intended audience

- The course instructor and evaluators reviewing the practical work.
- Students learning classical machine learning who want a worked, commented reference.
- Anyone who wants a compact tour of the standard scikit-learn workflow and an introductory Keras model.

### What makes it technically interesting

- **Breadth with a consistent method.** Instance-based, linear, tree-based, probabilistic, margin-based, ensemble and neural models are all evaluated with the same discipline: separate training, validation and test data, plus cross-validation where appropriate.
- **Realistic data problems.** Features on very different scales (credit limit), missing values, categorical encoding, class imbalance (customer churn), outliers (city survey) and overlapping classes are all handled explicitly rather than avoided.
- **Production-style tooling.** `Pipeline`, `ColumnTransformer`, custom transformers, `GridSearchCV` and `RandomizedSearchCV` are used so that preprocessing is fitted on training data only.
- **Original experiments.** Each lab contains a small, self-contained comparison (for example distance metrics, gini versus entropy, optimizers) that prints its own conclusion from measured numbers.

### Lab index

| Lab | Topic                                  | Datasets                                  | Original experiment                    |
| --- | -------------------------------------- | ----------------------------------------- | -------------------------------------- |
| 1   | k-Nearest Neighbours                   | Wine, synthetic sine wave                 | Comparison of three distance metrics   |
| 2   | Linear Regression                      | Diabetes, synthetic city wellbeing survey | Stability check over 20 random splits  |
| 3   | End-to-end ML project                  | Synthetic used cars, synthetic Pune flats | Baseline (dummy) model check           |
| 4   | k-NN hyper-parameter tuning            | Digits, synthetic credit limit            | Tuned k-NN versus tuned Decision Tree  |
| 5   | Decision Trees with Graphviz           | Wine, synthetic loan approval             | Gini versus entropy across depths      |
| 6   | Linear vs logistic regression vs trees | Synthetic, Diabetes, Breast Cancer        | Effect of regularisation strength `C`  |
| 7   | Support Vector Machines                | Wine, concentric circles                  | SVM versus k-NN under increasing noise |
| 8   | Naive Bayes                            | Wine, synthetic telecom churn             | Effect of `var_smoothing`              |
| 9   | Ensemble learning                      | Synthetic 2-D data, Breast Cancer         | Number of trees in a Random Forest     |
| 10  | Neural network (Keras)                 | Fashion-MNIST, Digits                     | Comparison of three optimizers         |

---

## 2. Tech Stack and Setup

### Tools actually used

| Area                        | Tools                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------- |
| Language                    | Python 3                                                                                |
| Numerical and data handling | NumPy, pandas, SciPy                                                                    |
| Machine learning            | scikit-learn (models, pipelines, cross-validation, grid and randomized search, metrics) |
| Deep learning               | TensorFlow / Keras (the notebook output records TensorFlow 2.21.0)                      |
| Visualisation               | Matplotlib, Seaborn, Graphviz (decision-tree rendering)                                 |
| Environment                 | Jupyter Notebook or JupyterLab; the notebook was executed on Google Colab               |

### Prerequisites

- Python 3.10 or newer.
- The Graphviz system program (not only the Python package) for Labs 5 and 6.
- Internet access on first run, because Fashion-MNIST is downloaded by Keras (approximately 30 MB). All other datasets ship with scikit-learn or are generated inside the notebook.
- Roughly 2 GB of free memory; no GPU is required.

### Installation

bash

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install numpy pandas scipy matplotlib seaborn scikit-learn graphviz tensorflow jupyterlab

# 4. Install the Graphviz system program (needed to render trees)
#    Ubuntu / Debian:  sudo apt-get install graphviz
#    macOS (Homebrew): brew install graphviz
#    Windows:          install from https://graphviz.org/download/ and add it to PATH
```

### Configuration

No API keys, credentials or configuration files are required. Random seeds are fixed inside each lab so that results are repeatable. The notebook writes a few small files to the working directory while it runs (for example `datasets/*.csv`, `telecom_churn.csv`, `wine_tree_d3.dot`, `loan_tree_pruned.png`); these can be deleted safely.

### Reproducing the results

bash

```bash
jupyter lab A049_Harshit_Jaiswal_sml_lab_journal.ipynb
```

In Jupyter choose **Kernel > Restart and Run All**. To run it non-interactively:

bash

```bash
jupyter nbconvert --to notebook --execute --inplace A049_Harshit_Jaiswal_sml_lab_journal.ipynb
```

Alternatively, upload the notebook to Google Colab; only the Graphviz system package may need installing there (`!apt-get install -y graphviz`).

---

## 3. Experiments and Evaluation Methodology

### Datasets

| Dataset                                          | Source                  | Size                                      | Used in         |
| ------------------------------------------------ | ----------------------- | ----------------------------------------- | --------------- |
| Wine                                             | scikit-learn, real      | 178 samples, 13 features, 3 classes       | Labs 1, 5, 7, 8 |
| Diabetes                                         | scikit-learn, real      | 442 patients, 10 features                 | Labs 2, 6       |
| Digits                                           | scikit-learn, real      | 1,797 images (8x8), 10 classes            | Labs 4, 10      |
| Breast Cancer Wisconsin                          | scikit-learn, real      | 569 samples, 30 features, 2 classes       | Labs 6, 9       |
| Fashion-MNIST                                    | Keras download, real    | 70,000 images (28x28), 10 classes         | Lab 10          |
| Sine-wave regression                             | Generated               | 100 samples, 1 feature                    | Lab 1           |
| City wellbeing survey and income                 | Generated, two files    | 35 cities                                 | Lab 2           |
| Used cars                                        | Generated               | 4,000 rows, 2% missing values             | Lab 3           |
| Pune flats                                       | Generated               | 3,500 rows, about 3% missing values       | Lab 3           |
| Credit limit                                     | Generated               | 1,200 customers, mixed scales             | Lab 4           |
| Loan approval                                    | Generated               | 1,500 applicants, 5% label noise          | Lab 5           |
| Telecom churn                                    | Generated               | 5,000 customers, about 13% positive class | Lab 8           |
| Circles, blobs, multi-class and 2-D cluster data | scikit-learn generators | 240 to 700 samples                        | Labs 6, 7, 9    |

Synthetic data is produced with seeded NumPy generators, so every run creates identical files.

### Baselines

- **Default models.** An untuned k-NN (Lab 4) is compared against the tuned pipeline.
- **Mean predictor.** A `DummyRegressor` that always predicts the training mean is used in Lab 3 to confirm the model has learned real structure.
- **Single weak learner.** A decision stump and a single unrestricted tree are the baselines for the ensemble methods in Lab 9.
- **Cross-lab model comparison.** Linear versus tree versus logistic models are compared on shared data and shared splits in Lab 6.

### Metrics

- Classification: accuracy, precision, recall, F1 score (weighted), confusion matrix, misclassification percentage.
- Regression: R-squared, RMSE, MSE.
- Model selection: mean and standard deviation of k-fold cross-validation scores; out-of-bag score for bagged ensembles.

### Test procedure

1. **Splits.** Labs use stratified train/test splits (25 to 30 percent test), a hash-based split on a stable identifier, or a 70/15/15 train/validation/test split (Lab 6).
2. **Preprocessing inside pipelines.** Imputation, scaling and encoding are fitted on training data only and merely applied to test data, which prevents leakage.
3. **Model selection.** Hyper-parameters are tuned with 5-fold cross-validation (`GridSearchCV`, `RandomizedSearchCV`) or on the validation split.
4. **Final evaluation.** The selected model is scored once on the held-out test set.

### Comparison conditions

Where two models or settings are compared, they use the same data, the same split and the same random seed. Each lab's original experiment changes exactly one factor (metric, criterion, regularisation, optimizer, number of trees, and so on) while holding the rest constant.

### Stated limitations of the method

- Test sets are small for several datasets (for example about 45 wines in Labs 1 and 5), so one prediction changes accuracy by roughly two percentage points.
- Most results come from a single seed; only Lab 2 repeats the split 20 times to quantify variability.
- Synthetic datasets were designed by the author and reflect the assumptions built into them rather than real-world behaviour.

---

## 4. Results and Analysis

All figures below are the measured outputs recorded in the notebook.

### Lab 1: k-Nearest Neighbours

| Experiment                               | Result                                            |
| ---------------------------------------- | ------------------------------------------------- |
| Wine, two features, k = 7                | Test accuracy 0.87                                |
| Wine, all 13 features, raw versus scaled | Best accuracy 0.822 raw, 1.000 scaled (k = 15)    |
| Sine-wave regression, k = 4              | Test R-squared 0.80; best k = 7 gives 0.856       |
| Distance metric (5-fold CV, scaled)      | Euclidean 0.967, Manhattan 0.955, Chebyshev 0.916 |

**Analysis.** Feature scaling produced the largest gain (0.822 to 1.000) because `proline` otherwise dominates the distance. Euclidean and Manhattan distance performed similarly, while Chebyshev was clearly weaker since it considers only the single largest feature difference.

### Lab 2: Linear Regression

| Experiment                       | Result                                                                        |
| -------------------------------- | ----------------------------------------------------------------------------- |
| Diabetes, BMI only               | Test R-squared 0.370, RMSE 61.3                                               |
| Diabetes, BMI + bp + s5 + s3     | Test R-squared 0.507, RMSE 54.2                                               |
| 20 random splits, mean R-squared | 0.328 (BMI only) versus 0.470 (multiple); multiple regression won 20 of 20    |
| City income model                | Training R-squared 0.928; Kolhapur predicted 4.89 (linear) versus 4.82 (5-NN) |

**Analysis.** The repeated-split experiment shows that the improvement from additional features is consistent and not a product of one favourable split. The two outlier cities visibly flatten the fitted line when they are not removed.

### Lab 3: End-to-end ML project

| Project                           | Model                                                      | Result                                                                         |
| --------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Used cars (5-fold CV RMSE, lakhs) | Linear Regression / Decision Tree / Random Forest          | 1.094 / 1.027 / 0.752                                                          |
| Used cars (final)                 | Tuned Random Forest                                        | Test RMSE 0.733 lakhs, R-squared 0.933; error 74% lower than the mean baseline |
| Pune flats (CV RMSE, lakhs)       | Gradient Boosting / Random Forest / Linear / Decision Tree | 15.008 / 15.972 / 18.727 / 21.638                                              |
| Pune flats (final)                | Gradient Boosting                                          | Test RMSE 15.051 lakhs, R-squared 0.937                                        |

**Analysis.** The unrestricted Decision Tree has near-zero training error and the worst cross-validated error, a clear case of overfitting. Linear Regression has similar training and CV error but a higher overall error, indicating underfitting of multiplicative effects such as area times locality rate. Stratified sampling reduced sampling bias in the area categories compared with a plain random split.

### Lab 4: k-NN hyper-parameter tuning

| Experiment                            | Result                                                                                                |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Digits, best grid-search pipeline     | k = 3, Euclidean, no scaler, 40 PCA components; CV accuracy 0.9865                                    |
| Digits test accuracy                  | 0.9907 tuned versus 0.9889 untuned                                                                    |
| Credit limit, k = 5                   | Test R-squared 0.657 unscaled versus 0.883 scaled                                                     |
| Credit limit grid search              | Best: k = 6, distance weights, RobustScaler; CV R-squared 0.900, test R-squared 0.905, RMSE Rs 36,591 |
| Randomized versus grid search         | 125 fits in 1.2 s versus 1,500 fits in 17.5 s; CV R-squared 0.891 versus 0.900                        |
| Tuned k-NN versus tuned Decision Tree | Test R-squared 0.905 versus 0.878                                                                     |

**Analysis.** Scaling matters enormously when feature units differ. On the easy Digits problem tuning changes the test score by only a couple of images, so its main value is a trustworthy cross-validated estimate and a more compact model. Randomized search recovered most of the grid-search quality at a fraction of the cost.

### Lab 5: Decision trees

| Experiment                       | Result                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------- |
| Wine, entropy, depth 3           | Test accuracy 0.933 (identical to the unrestricted tree)                        |
| Loan approval, pruned tree       | 14 leaves, test accuracy 0.936                                                  |
| Loan approval, unrestricted tree | 107 leaves, training accuracy 1.000, test accuracy 0.878                        |
| Gini versus entropy              | Mean CV accuracy 0.914 versus 0.907; best single result gini at depth 3 (0.927) |

**Analysis.** Pruning removed over-fitting to the 5% noisy labels and improved test accuracy while keeping the tree readable. Criterion choice mattered far less than tree depth.

### Lab 6: Linear, logistic and tree models

| Task                                           | Result                                                                 |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| Synthetic regression (sine plus trend)         | Linear test MSE 1.460 versus Decision Tree (depth 9) 0.812             |
| Diabetes regression                            | Linear test MSE 5,594 versus Decision Tree (depth 1) 5,938             |
| Synthetic logistic regression                  | Test misclassification 16.7%                                           |
| Breast Cancer, two-feature logistic regression | Test misclassification 5.8%                                            |
| Breast Cancer tree, full versus depth 4        | Test error 5.81% (17 leaves) versus 3.49% (11 leaves)                  |
| Effect of `C`                                  | Validation error 10.59% at `C` = 0.001 and 9.41% from `C` = 0.1 upward |

**Analysis.** A tree captures non-linear structure that a line cannot, but on noisy real data (Diabetes) it offers no advantage. Over-strong regularisation (very small `C`) increases error; beyond a moderate value the results are flat.

### Lab 7: Support Vector Machines

| Experiment                                           | Result                |
| ---------------------------------------------------- | --------------------- |
| Linear SVM, Wine class_0 versus rest                 | Test accuracy 0.963   |
| Wine, all features: linear / RBF / polynomial kernel | 0.963 / 0.981 / 0.963 |
| RBF without scaling                                  | 0.704                 |

| Noise level | SVM (RBF) | k-NN (k = 7) |
| ----------- | --------- | ------------ |
| 0.05        | 1.000     | 1.000        |
| 0.15        | 0.950     | 0.950        |
| 0.30        | 0.842     | 0.817        |
| 0.45        | 0.708     | 0.692        |

**Analysis.** Scaling is mandatory for kernel SVMs (0.981 to 0.704 without it). The SVM was equal to or slightly better than k-NN at every noise level, and both degrade as the rings overlap.

### Lab 8: Naive Bayes

| Experiment                            | Result                                                    |
| ------------------------------------- | --------------------------------------------------------- |
| Wine, Gaussian NB                     | Accuracy 0.944, weighted F1 0.944                         |
| Churn, learned priors                 | Accuracy 0.864, churn precision 0.435, churn recall 0.050 |
| Churn, equal priors                   | Accuracy 0.620, churn precision 0.218, churn recall 0.711 |
| `var_smoothing` 1e-9 / 1e-3 / 1 / 100 | CV accuracy 0.966 / 0.764 / 0.646 / 0.399                 |

Decision-threshold tuning on the churn model:

| Threshold | Accuracy | Churn precision | Churn recall | Customers flagged |
| --------- | -------- | --------------- | ------------ | ----------------- |
| 0.50      | 0.864    | 0.435           | 0.050        | 23                |
| 0.35      | 0.848    | 0.374           | 0.199        | 107               |
| 0.25      | 0.784    | 0.290           | 0.423        | 293               |
| 0.18      | 0.699    | 0.246           | 0.602        | 492               |
| 0.12      | 0.586    | 0.210           | 0.756        | 724               |

**Analysis.** On imbalanced data, accuracy hides the real behaviour. Lowering the threshold trades precision and accuracy for recall of the minority class; the appropriate operating point depends on the cost of missing a churner versus contacting a loyal customer.

### Lab 9: Ensemble learning

| Method                | Part A (synthetic 2-D) | Part B (Breast Cancer)    |
| --------------------- | ---------------------- | ------------------------- |
| Single decision stump | 0.851                  | 0.902                     |
| Single decision tree  | 0.834                  | 0.930                     |
| Voting (hard / soft)  | 0.897 / 0.897          | 0.972 / 0.972             |
| Bagging               | 0.891 (OOB 0.922)      | 0.958 (OOB 0.953)         |
| Random Forest         | 0.880                  | 0.951                     |
| AdaBoost (150 stumps) | 0.903                  | 0.972                     |
| Best individual model | SVC 0.909              | Logistic Regression 0.979 |

Random Forest size experiment (Breast Cancer): out-of-bag accuracy rose from 0.892 with 5 trees to 0.965 with 400 trees; test accuracy changed by only +0.014 over the same range.

**Analysis.** Ensembles clearly beat a single tree and a single stump. On Breast Cancer a scaled Logistic Regression is already very strong, so ensembles match rather than exceed it. Forest accuracy flattens after a few dozen trees, and the out-of-bag score tracks test accuracy closely without using the test set.

### Lab 10: Neural network with Keras

| Experiment                                                          | Result                                                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Fashion-MNIST, Dense(256) + softmax, 203,530 parameters, 6 epochs   | Test accuracy 0.8672, test loss 0.3682                                         |
| Digits, Dense(48) + Dropout + Dense(24) + softmax, 4,546 parameters | Stopped by early stopping at 81 epochs; test accuracy 0.9733, test loss 0.0845 |

| Optimizer (Digits, 30 epochs) | Final validation accuracy | Test accuracy |
| ----------------------------- | ------------------------- | ------------- |
| adam                          | 0.959                     | 0.973         |
| rmsprop                       | 0.959                     | 0.973         |
| sgd                           | 0.874                     | 0.911         |

**Analysis.** Fashion-MNIST is harder than digit recognition because several clothing classes look alike. On the small Digits set, Dropout and early stopping controlled overfitting. Adaptive optimizers learned much faster than plain SGD within the same epoch budget.

### Runtime and cost

All experiments run on a CPU. Representative timings recorded in the notebook: Digits grid search (1,260 fits) 65.4 s, credit-limit grid search (1,500 fits) 17.5 s, randomized search 1.2 s, Fashion-MNIST training about 2 to 4 s per epoch. The project requires no paid services; the free tier of Google Colab or a standard laptop is sufficient.

---

## 5. Failure Analysis and Limitations

| Observed failure                                                         | Root cause                                                                                           | Mitigation tried                                                  | Status                                                                            |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| k-NN on unscaled Wine reaches only 0.822; on credit data R-squared 0.657 | Large-valued features (`proline`, income) dominate Euclidean distance                                | `StandardScaler`, `MinMaxScaler`, `RobustScaler` inside pipelines | Resolved (0.883 on credit data)                                                   |
| RBF-SVM accuracy falls to 0.704 without scaling                          | Kernel distance dominated by one feature                                                             | Scaling in a pipeline                                             | Resolved                                                                          |
| Unrestricted decision trees overfit (107 leaves, 0.878 test accuracy)    | Tree memorises the noisy labels                                                                      | `max_depth`, `min_samples_leaf` pruning                           | Resolved                                                                          |
| Linear models underfit the sine data and the circles dataset             | Boundary or trend is non-linear                                                                      | Regression trees, polynomial features, RBF and polynomial kernels | Resolved for these datasets                                                       |
| Naive Bayes churn recall is only 0.05                                    | 13% positive class and the learned prior favours "Stayed"; independence assumption is violated       | Equal priors (recall 0.711), threshold tuning                     | Partly resolved: higher recall costs precision and accuracy                       |
| Equal priors drop accuracy to 0.620                                      | Many loyal customers are flagged                                                                     | Threshold table to choose an operating point                      | Open: depends on business cost                                                    |
| Very large `var_smoothing` collapses accuracy to 0.399                   | Variances become so wide that classes are indistinguishable                                          | Keep the default or a very small value                            | Resolved                                                                          |
| Plain SGD lags (0.911)                                                   | Fixed step size converges slowly in 30 epochs                                                        | Compare adaptive optimizers                                       | Resolved by choosing adam or rmsprop; more epochs or tuned learning rate untested |
| Fashion-MNIST accuracy limited to 0.867                                  | Dense network ignores image structure; similar classes (shirt, T-shirt, pullover, coat) are confused | None beyond the baseline architecture                             | Open: convolutional layers are the natural fix                                    |

### Known limitations

- **Selection on the test set.** In a few exploratory cells (for example the best k in Lab 1, best boosting round in Lab 9) the best setting is read from test accuracy for illustration. Those numbers are optimistic and should not be quoted as unbiased estimates.
- **Small samples.** Some test sets contain fewer than 60 examples, so differences of one or two predictions are noise.
- **Synthetic data.** Synthetic datasets follow formulas chosen by the author, so conclusions about, for example, house or car prices do not transfer to real markets.
- **Single run.** Except where stated, scores come from one random seed and one split, without confidence intervals.
- **Environment drift.** Different library versions can change numeric results slightly, particularly cross-validation scores and neural-network training.

### Security considerations

- The notebook handles no personal data; all customer, loan and housing records are synthetic.
- No credentials or API keys are used or stored.
- The only network access is the Fashion-MNIST download from the official Keras dataset host. Dependencies should be installed from the official package index in an isolated virtual environment, and pinned to exact versions for any shared deployment.
- Notebooks can execute arbitrary code. Open notebooks from untrusted sources only in a sandbox or with the "Not Trusted" state preserved.
- Generated files are written to the working directory; run the notebook in a dedicated folder.

---

## 6. Testing, Roadmap and References

### Testing

The project has no separate unit-test suite; correctness is checked by execution and by built-in assertions of behaviour:

1. Run **Restart and Run All** (or the `nbconvert --execute` command above). A clean run with no error cells is the primary check.
2. Confirm that the headline figures match Section 4. Because seeds are fixed, classical-model results should reproduce exactly on the same library versions; neural-network results may vary slightly.
3. Sanity checks are embedded in the notebook: for example the imputer statistics are compared with pandas medians, stratified proportions are compared with the full data, and each original experiment prints a data-derived conclusion.
4. To guard against regressions after editing, run the notebook inside a clean virtual environment and compare the printed summary tables against this document.

### Roadmap

- Add an `requirements.txt` with pinned versions and a Dockerfile for exact reproducibility.
- Add automated execution of the notebook in continuous integration (for example GitHub Actions with `nbconvert`).
- Repeat key experiments over multiple seeds and report mean and standard deviation or confidence intervals.
- Replace test-set selection in exploratory cells with a nested cross-validation or a separate validation split.
- Extend Lab 10 with a convolutional network on Fashion-MNIST and a learning-rate comparison for SGD.
- Add probability calibration and cost-sensitive evaluation for the churn problem.
- Evaluate the models on additional real-world datasets in place of synthetic ones.

### References

- Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830, 2011.
- Abadi, M. et al. "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems." 2015. Keras documentation: [https://keras.io](https://keras.io)
- Geron, A. *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*. O'Reilly Media. (End-to-end project workflow in Lab 3 follows this structure.)
- Cover, T. and Hart, P. "Nearest neighbor pattern classification." *IEEE Transactions on Information Theory*, 1967.
- Cortes, C. and Vapnik, V. "Support-vector networks." *Machine Learning*, 20, 1995.
- Breiman, L. "Random forests." *Machine Learning*, 45, 2001.
- Freund, Y. and Schapire, R. "A decision-theoretic generalization of on-line learning and an application to boosting." *Journal of Computer and System Sciences*, 55, 1997.
- Kingma, D. and Ba, J. "Adam: A method for stochastic optimization." 2015.
- Xiao, H., Rasul, K. and Vollgraf, R. "Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms." 2017.
- Wine, Diabetes, Digits and Breast Cancer Wisconsin datasets as distributed with scikit-learn (originating from the UCI Machine Learning Repository and the sources cited in their scikit-learn documentation).
- Graphviz: [https://graphviz.org](https://graphviz.org)

---

## 7. Acknowledgement

I sincerely thank **Prof. Pratik A. Desai** for teaching all of the laboratories and practicals in Semester 1 on which this journal is based. His clear explanations, structured exercises and encouragement to experiment beyond the prescribed problems made it possible to consolidate the entire course into this single notebook.

I also acknowledge the open-source communities behind scikit-learn, TensorFlow and Keras, NumPy, pandas, Matplotlib and Graphviz, and the creators of the public datasets used throughout this work.

*Harshit Jaiswal, Roll No. A049*
