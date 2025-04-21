![MKDPINN: Meta Learning and Knowledge Discovery-Based Physics-Informed Neural Networks](img%2Fbanner.jpeg)

# MKDPINN: Meta Learning and Knowledge Discovery-Based Physics-Informed Neural Networks


<p align="center">
  <a href="https://github.com/Sephiroth66616/MKDPINN">
    <img src="https://img.shields.io/badge/Platform-PyTorch-red" alt="Platform: PyTorch">
  </a>
  <a href="https://github.com/Sephiroth66616/MKDPINN">
    <img src="https://img.shields.io/badge/Language-Python-blue" alt="Language: Python">
  </a>
  <a href="https://github.com/Sephiroth66616/MKDPINN/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/Sephiroth66616/MKDPINN">
    <img src="https://img.shields.io/github/stars/Sephiroth66616/MKDPINN?style=social" alt="GitHub Stars">
  </a>
</p>


**MKDPINN** is a novel approach for Remaining Useful Life (RUL) prediction, combining meta-learning, knowledge discovery, and Physics-Informed Neural Networks (PINNs). This repository contains the code for the paper "*Meta-Learning and Knowledge Discovery based Physics-Informed Neural Network for Remaining Useful Life Prediction of Rotating Machinery*".

## Overview

Traditional data-driven methods for RUL prediction often struggle with limited data and generalization to new operating conditions. MKDPINN addresses these challenges by:

1.  **Meta-Learning:** Learns from multiple related tasks (e.g., different machines or operating conditions) to enable fast adaptation to new tasks with limited data.
2.  **Knowledge Discovery:** Extracts underlying physical relationships (expressed as partial differential equations or PDEs) from data, improving model interpretability and accuracy.
3.  **Physics-Informed Neural Networks (PINNs):** Incorporates physical knowledge (PDEs) into the neural network training process, guiding the learning process and improving generalization.

## Key Features

*   **Meta-Learning Framework:** Uses the first order meta learning algorithm for efficient meta-learning.
*   **Hidden State Maper (HSM):** Extracts a hidden state representation from sensor data.
*   **Predictor:** Predicts RUL based on the hidden state and time.
*   **Physics-Guided Regularizer (PGR):** Discovers and enforces underlying PDE constraints.
*   **0-Shot Learning:** Designed for scenarios where training and testing data come from the same task (e.g., the same machine).  Supports 0-shot testing *only* for the provided dataset.
*   **Early Stopping:** Prevents overfitting during training.
*   **Clear Logging:**  Provides detailed training progress and results.

## Repository Structure

```
MKDPINN/
├── LICENSE
├── README.md
├── requirements.txt
├── main.py          (Main script for training and testing)
├── data/            (Placeholder for datasets - see instructions below)
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── components.py  (HSM, PREDICTOR, PGR components)
│   ├── mkdpinn.py   (MKDPINN model definition)
│   ├── preprocessing.py (Data loading and preprocessing)
│   └── utils/
│       ├── __init__.py
│       └── early_stopping.py (EarlyStopping class)
└── results/          (Trained models and logs will be saved here)
```

# Attempting to call train() will display:
```python
"Training code reserved for peer review. Full implementation will be released upon paper acceptance."
```
## Getting Started

### 1. Prerequisites

*   Python 3.7+
*   PyTorch 1.13+ (with CUDA support if using a GPU)
*   NumPy, SciPy, Matplotlib, tqdm

### 2. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Sephiroth66616/MKDPINN.git 
    cd MKDPINN
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # (Linux/macOS)
    .venv\Scripts\activate  # (Windows)
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


### 3. Data

*   This repository is designed to work with the **FD004** subset of the C-MAPSS dataset.
*   **Download the C-MAPSS dataset:** You can obtain the C-MAPSS dataset from the NASA Prognostics Center of Excellence ([https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)).  Specifically, download the `FD004` dataset.
*   **Place the data:**  Put the `FD004` data files (`train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`) into the `data/` directory.  The `preprocessing.py` script assumes this location.

### 4. Training

1.  **Run `main.py`:**

    ```bash
    python main.py
    ```

2.  **Configuration:**
    *   You can adjust training parameters (e.g., learning rate, hidden dimensions, meta-learning parameters) in the `main.py` file.
    *   The `meta_param` list in `main.py` controls the meta-learning hyperparameters:
        *   `[learning_rate, inner_steps, outer_step_size, outer_iterations, inner_batch_size, meta_batch_size, n_shot]`

3.  **Logging:**
    *   Training progress and results will be printed to the console and saved in a `training.log` file in the project's root directory.
    *   Trained model checkpoints (for HSM, Predictor, and PGR) will be saved in the `results/` directory.

### 5. Testing

*   **Zero-Shot Testing:** The provided code is configured for 0-shot testing on FD004, meaning the test data comes from the same engines as the training data.
*   **Testing during training:**  The `test` function is called after each epoch during training to evaluate the model's performance on the test set.
*  **Training model:**  Uncomment `mkdpinn.train` in the `main.py` file

### 6. Important Notes

*   **0-Shot Only:** This implementation is *specifically designed* for 0-shot testing on the provided dataset.  Setting `n_shot` to a value other than 0 will *not* enable few-shot testing; it will still perform 0-shot testing and print a warning message.
*   **Dataset:** The code *requires* the FD004 dataset to be placed in the `data/` directory.

### 7. Experimental Results

The following table compares the performance of MKDPINN with recently published models on the C-MAPSS dataset.  Metrics include Root Mean Squared Error (RMSE) and Score. Lower values indicate better performance.
## Experimental Results

The following table compares the performance of MKDPINN with recently published models on the C-MAPSS dataset. Metrics include Root Mean Squared Error (RMSE) and Score. Lower values indicate better performance.

| Model                                                                        | FD001       |          | FD002       |          | FD003       |          | FD004       |          | Average     |             |
| :--------------------------------------------------------------------------- | :---------- | :------- | :---------- | :------- | :---------- | :------- | :---------- | :------- | :---------- | :---------- |
|                                                                              | RMSE        | SCORE    | RMSE        | SCORE    | RMSE        | SCORE    | RMSE        | SCORE    | RMSE        | SCORE       |
| [RVE](https://doi.org/10.1016/j.ress.2022.108353) (2022)                       | 13.42       | 323.82   | 14.92       | 1379.17  | 12.51       | 256.36   | 16.37       | 1845.99  | 14.31       | 951.34      |
| [ED-LSTM](https://doi.org/10.1016/j.ress.2023.109666) (2023)                    | **9.14**    | **53**   | 18.17       | 1693     | 11.96       | 238      | 18.51       | 2160     | 14.45       | 1036.00     |
| [AttnPINN](https://doi.org/10.1016/j.aei.2023.102195) (2023)                     | 16.89       | 523      | 16.32       | 1479     | 17.75       | 1194     | 18.37       | 2059     | 17.33       | 1313.75     |
| [MSTSDN](https://doi.org/10.1021/acsomega.4c03873) (2024)                       | 13.67       | 246.01   | 16.28       | 1342.59  | 13.66       | 258.92   | 17.33       | 1641.51  | 15.24       | 872.26      |
| [ARR](https://doi.org/10.1016/j.engappai.2024.108475) (2024)                     | 11.36          |  192.22       |   18.97          |    2433.15      |   **11.28**       |  **133.41**        |     20.69        |    2842.44      |   15.58       |    1400.31     |
| [Meta-Transformer](https://doi.org/10.1109/PHM-Beijing63284.2024.10874723) (2024) | 12.28       | /        | 14.49       | /        | 12.86       | /        | 15.90       | /        | 13.88       | /           |
| [MGCAL-UQ](https://doi.org/10.1007/s40430-025-05400-8) (2025)                   | 11.63       | 180.61   | 17.16       | 1481.92  | 12.42       | 230.28   | 16.10       | 1278.56  | 14.33       | 792.84      |
| [DAM](https://doi.org/10.3390/s25020497) (2025)                                 | 13.03       |   217    |     15.41        |    **796**       |   12.21         |   189      |     16.43        |    1029      |   14.27       |    557.75     |
| MKDPINN (proposed)                                                              | 12.36       | 273.19   | **13.85**   | 1037.75 | 11.01 | 221.24   | **13.60**   | **956.42** | **12.71**   | **622.15**  |

**Note:** The best results for each dataset are highlighted in **bold**. A "/" indicates that the corresponding result was not reported in the original paper.  Average values are calculated across all four datasets where data is available.


## Citation

If you use this code in your research, please cite our paper:

```
@article{wang2025mkdpinn,
  title     = {Meta-Learning and Knowledge Discovery based Physics-Informed Neural Network for Remaining Useful Life Prediction},
  author    = {Wang, Yu and Liu, Shujie and Lv, Shuai and Liu, Gengshuo},
  journal   = {arXiv preprint arXiv:2504.13797},
  year      = {2025},
  eprint    = {2504.13797},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi       = {10.48550/arXiv.2504.13797},
  url       = {https://arxiv.org/abs/2504.13797}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   The C-MAPSS dataset was provided by the NASA Prognostics Center of Excellence.

---