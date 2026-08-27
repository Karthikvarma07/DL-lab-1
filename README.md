# Single Layer Perceptron — Banknote Authentication

A from-scratch implementation of a single-layer perceptron (built with NumPy, no ML framework) trained to classify banknotes as **authentic** or **forged** using the UCI Banknote Authentication dataset.

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Banknote Authentication Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt)
- **Samples:** 1372
- **Features:** 4 numeric features extracted from wavelet-transformed banknote images
  - `Variance`
  - `Skewness`
  - `Curtosis`
  - `Entropy`
- **Target:** `Class` — `0` (Authentic) or `1` (Forged)

The dataset is loaded directly from the URL, so no local data file is needed.

## 🧠 What this notebook does

1. **Exploratory Data Analysis (EDA)**pdf
   - Inspects shape, dtypes, and missing values
   - Feature histograms, correlation heatmap, scatter plot, and boxplots

2. **Preprocessing**
   - Standardizes features with `StandardScaler`
   - Splits data into training/testing sets (80/20)

3. **Perceptron implementation (from scratch)**
   - Step activation function
   - Manual weight/bias initialization and update rule (Perceptron Learning Algorithm)
   - Trains for 20 epochs, tracking misclassifications, weights, and bias per epoch

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix (with heatmap visualization)
   - Training error curve across epochs
   - Weight and bias evolution plots

5. **Learning rate comparison**
   - Trains the perceptron with learning rates `0.001`, `0.01`, and `0.1` to compare convergence behavior

6. **Summary table**
   - Consolidated results (dataset size, split ratio, hyperparameters, final weights/bias, and performance metrics)

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## ▶️ Usage

1. Clone this repository
2. Install the dependencies above
3. Open and run the notebook:

```bash
jupyter notebook singlelayerperceptron_3_.ipynb
```

All cells run top to bottom; the dataset is fetched automatically from the UCI repository URL.

## 📈 Results

The notebook prints and visualizes:
- Per-epoch misclassification counts
- Final accuracy/precision/recall/F1-score on the test set
- A confusion matrix distinguishing authentic vs. forged notes
- Convergence comparison across different learning rates

*(Exact metric values depend on the run since the perceptron trains from scratch each time.)*

## 📁 Project Structure

```
.
├── singlelayerperceptron_3_.ipynb   # Main notebook
└── README.md
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
