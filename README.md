# DeepLearners Scientific Computing Toolkit

The **DeepLearners Toolkit** is a modular scientific computing and machine learning utility package built entirely using **Python + NumPy**.  
It is developed as part of the University of Adelaide’s *Scientific Computing & Programming* course and demonstrates clean software architecture, numerical computing fundamentals, preprocessing, ranking, metrics, optimisation, pipelines, and benchmarking.

This repository contains multiple standalone modules that together form a lightweight, educational alternative to common scientific computing libraries.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Sammie1875/deeplearners-toolkit.git
cd deeplearners-toolkit

Install dependencies:

python -m pip install numpy pytest

📁 Project Structure

deeplearners-toolkit/
│
├── deeplearners_toolkit/
│   ├── io.py                 # Data loading
│   ├── preprocessing.py      # Scaling & encoding
│   ├── sort_search.py        # Sorting, searching, Top‑K
│   ├── rank.py               # Ranking & percentiles
│   ├── stats.py              # Statistics
│   ├── metrics.py            # Classification & regression metrics
│   ├── optim.py              # Gradients & Jacobians
│   ├── pipeline.py           # Transformer/Estimator pipeline
│   ├── utils.py              # Shared utilities
│   ├── benchmarking.py       # Vectorised vs loop benchmarking
│
├── tests/                    # Unit tests
├── demo/quickstart.ipynb     # End‑to‑end demonstration
└── README.md

🚀 Quickstart Example

import numpy as np

from deeplearners_toolkit.io import read_csv
from deeplearners_toolkit.preprocessing import StandardScaler
from deeplearners_toolkit.stats import mean
from deeplearners_toolkit.sort_search import top_k
from deeplearners_toolkit.rank import rankdata
from deeplearners_toolkit.metrics import accuracy
from deeplearners_toolkit.optim import gradient
from deeplearners_toolkit.pipeline import Pipeline

# Load data
data = read_csv("data.csv")

# Preprocess (example)
scaler = StandardScaler()
scaled = scaler.fit_transform(data)

# Statistics
print("Mean:", mean(scaled))

# Sorting / Top‑K
print("Top‑3:", top_k(scaled[:, 0], k=3))

# Ranking
print("Ranks:", rankdata(scaled[:, 0]))

# Metrics (example)
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([1, 0, 0, 1])
print("Accuracy:", accuracy(y_true, y_pred))

# Gradient (finite difference)
f = lambda x: x**2
print("Gradient at x=3:", gradient(f, 3.0))

# Pipeline chaining
pipeline = Pipeline([StandardScaler()])
output = pipeline.fit_transform(data)
print("Pipeline output:", output)



🧪 Running Tests
python -m pytest tests/


👥 Team Responsibilities

Sameena Shafiulla Khan
io.py
preprocessing.py
Unit tests
README + documentation

Veeksha
stats.py
metrics.py
benchmarking.py
Benchmark results

Krishna
rank.py
utils.py
demo/quickstart.ipynb

Aakash
sort_search.py
optim.py
pipeline.py
Integration + packaging

📄 Documentation
Full project documentation is available in:

docs/project_documentation.md

This includes detailed module explanations, algorithms, design decisions, examples, benchmarking results, and testing strategy.

🎓 Assignment Deliverables
Full toolkit implementation
≥ 20 unit tests (team target: 30+)
quickstart notebook
Benchmark results
README
Documentation
4‑page report
5‑minute demo video

📄 License
This project is for educational use as part of the University of Adelaide Scientific Computing assignment.






