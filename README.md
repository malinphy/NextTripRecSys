# Next Trip Recommendation System

This project implements a deep learning-based recommendation system to predict the next destination city in a trip sequence. It models sequential data using a **Residual Network** architecture with Entity Embeddings for cities and countries, implemented in **PyTorch**.

The project is designed to solve sequential recommendation problems using real-world travel data provided by Booking.com.

## � Dataset: Booking.com Multi-Destination Trips

This project utilizes the **Booking.com Multi-Destination Trips Dataset**, which contains millions of anonymized real hotel reservations.

> **Intro**: Many travelers go on trips which include more than one destination. Our mission at Booking.com is to make it easier for everyone to experience the world, and we can help to do that by providing real-time recommendations for what their next in-trip destination will be. By making accurate predictions, we help deliver a frictionless trip planning experience.

### Features
The dataset includes over a million (1,166,835) anonymized hotel reservations with the following features:
- `user_id`: User ID
- `checkin`: Reservation check-in date
- `checkout`: Reservation check-out date
- `created_date`: Date when the reservation was made
- `affiliate_id`: Anonymized ID of affiliate channels
- `device_class`: Desktop/Mobile
- `booker_country`: Country from which the reservation was made (anonymized)
- `hotel_country`: Country of the hotel (anonymized)
- `city_id`: City ID of the hotel's city (anonymized)
- `utrip_id`: Unique identification of user's trip (a group of multi-destinations bookings within the same trip).

### Citation
If you use this dataset or approach in your research, please cite the original resource paper:

*Dmitri Goldenberg and Pavel Levin. 2021. Booking.com Multi-Destination Trips Dataset. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21), July 11–15, 2021, Virtual Event, Canada.*

```bibtex
@inproceedings{goldenberg2021dataset,
 author =    {Goldenberg, Dmitri  and Levin, Pavel},
 title =     {Booking.com Multi-Destination Trips Dataset},
 booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21)},
 year =      {2021},
 doi  =      {10.1145/3404835.3463240}}
```

## �📌 Technical Features

- **Sequential Modeling**: Uses a window of previous visits (default: 5) to predict the next destination using numerical and categorical inputs.
- **Entity Embeddings**: Learns dense 128-dim embeddings for Cities and 64-dim embeddings for Countries.
- **Residual Architecture**: Deep network with Residual Blocks to effectively capture complex sequential patterns.
- **Modular Design**: Clean separation of concerns (Data, Model, Training, Inference).
- **Inference Ready**: Includes a specialized `Predictor` class and demo script for offline inference.

## 🧹 Feature Engineering & Data Preparation

This project uses `src/data/feature_engineering.py` to prepare, clean, and engineer features from the Booking.com dataset.

### 1. Data Sources & Merging
The script processes the following raw files:
- `train_set.csv`: Training data.
- `test_set.csv`: Test data (Last destinations are concealed with `city_id=0`).
- `ground_truth.csv`: The hidden targets for the test set.

**Critical Step: Test Set Completion**
The hidden final destinations (`city_id=0`) in the test set are filled using `ground_truth.csv`.
> **Goal:** To allow the model to see the complete history (up to the window size) when predicting the final destination of a trip.

### 2. Transformations & Encoding

**Label Encoding (Categorical)**
A global encoder is trained on the combined dataset (`train` + `test` + `ground_truth`).
- **Features:** `city_id`, `hotel_country`, `device_class`, `booker_country`, `affiliate_id`.
- **Shift (+1):** All encoded values form a 1-based index.
- **Padding (0):** The index `0` is reserved for padding in trips shorter than the window size.

**Derived Features**
For each reservation, the following features are generated:
- **Sequential**: `pos` (Step number), `days_from_start` (Cumulative days).
- **Temporal**: `stay_duration`, `gap_duration` (Days since last checkout), `checkin_dow` (Day of week), `checkin_month`.
- **Geographical**: `country_change` (Binary), `is_revisit` (Binary).

### 3. Sliding Window Method
Data is split into sliding windows of size **N=5**.
- **Input (X):** Sequence of 5 previous cities/countries (Left-padded with 0 if needed).
- **Target (y):** The 6th city in the sequence.

**Window Aggregates**
Statistical features calculated for each window:
1. `avg_stay_duration`: Average stay length.
2. `unique_countries_count`: Diversity of countries visited.
3. `avg_gap_duration`: Average time between bookings.
4. `total_days_so_far`: Total trip duration up to the current step.

### 4. Output Files
The processed data is saved to `processed_data/`:

- **`train_fe_eng.csv`**: The main training set. Contains:
  - All windows from `train_set.csv`.
  - **Intermediate windows** from `test_set.csv` (Trips' middle steps where the target is not hidden), increasing training data without leakage.
- **`final_test.csv`**: The evaluation set. Contains:
  - Only the **LAST** window of each trip in `test_set.csv`.
  - The `target_city` is the ground truth.

## 📂 Project Structure

```text
NextTripRecSys/
├── checkpoints/           # Saved models and embeddings
├── original_data/         # Raw data (train_set.csv, etc.)
├── processed_data/        # Processed data (train_fe_eng.csv, etc.)
├── src/
│   ├── data/              # Data processing pipelines
│   │   ├── feature_engineering.py  # Raw data -> Features (preprocessing logic)
│   │   ├── preprocessing.py        # Scaling & ID mapping
│   │   └── dataset.py              # PyTorch Dataset
│   ├── models/            # Model architecture (Residual Net)
│   ├── training/          # Training loop & logic
│   ├── evaluation/        # Metrics (Precision@K)
│   ├── inference/         # Prediction logic
│   └── config.py          # Central configuration
├── train.py               # Main training script
├── demo.py                # Inference demo script
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## 🚀 Installation

1. Clone the repository.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 🛠 Usage

### 1. Data Preparation
Ensure you have the following files in the `original_data/` directory:
- `train_set.csv`
- `test_set.csv`
- `ground_truth.csv`

Run the feature engineering script to preprocess and generate the training data:

```bash
python -m src.data.feature_engineering
```
This generates `train_fe_eng.csv` and `final_test.csv` in `processed_data/`.

### 2. Training
Train the model using the prepared data. Configurations can be adjusted in `src/config.py`.

```bash
python train.py
```
This will:
- Load processed data.
- Train the `V2ModelResidual`.
- Save model checkpoints and embeddings to `checkpoints/`.

### 3. Demo / Inference
After training, run the demo script to test the model on a random sample:

```bash
python demo.py
```


## 🏆 Evaluation Results
The model performance is evaluated using the **Precision@4** metric.

**Current Best Performance:**
- **Best Precision@4**: 0.5134 (at Epoch 6)

## ⚖️ Terms and Conditions
- The dataset is a property of Booking.com and may not be reused for commercial purposes.
- It may not be used in a manner that is harmful or competitive with Booking.com or Booking Holdings.
- It may not be used in any manner that violates any law or regulation.

For more details, visit the [Booking.com challenge website](https://www.bookingchallenge.com/) and [Booking.ai blog](https://booking.ai/).
