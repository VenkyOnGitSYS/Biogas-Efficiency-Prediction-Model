# 🔋 Biogas Efficiency Prediction Model

This project builds a **Biogas Efficiency Prediction Model** using a
**Random Forest Regressor** trained on sensor data (temperature,
humidity, and methane levels).

The model computes an efficiency score based on optimal anaerobic
digestion conditions and then trains a machine learning model to predict
efficiency from sensor readings.

------------------------------------------------------------------------

## 📌 Project Overview

Biogas production efficiency depends on environmental and gas parameters
such as:

-   🌡 **Temperature**
-   💧 **Humidity**
-   🔥 **Methane concentration (MQ2 sensor value)**

This project:

1.  Loads sensor data from a CSV file.
2.  Computes an efficiency score using a Gaussian scoring method.
3.  Trains a Random Forest regression model.
4.  Evaluates model performance using R² and MSE.
5.  Saves the trained model as a `.pkl` file.

------------------------------------------------------------------------

## 🧠 Efficiency Calculation Logic

Efficiency is computed using a **Gaussian scoring function** centered
around optimal values for anaerobic digestion.

### Optimal Values Used:

  Parameter       Optimal Value   Spread
  --------------- --------------- --------
  Temperature     37.5 °C         3
  Humidity        60 %            10
  Methane (MQ2)   800             150

### Formula

Each parameter is scored using:

exp(-((value - optimal)\^2) / (2 \* spread\^2))

Final Efficiency:

Efficiency = (Temp_score \* 0.4 + Humidity_score \* 0.3 + Methane_score
\* 0.3) \* 100

Efficiency is returned as a percentage (0--100%).

------------------------------------------------------------------------

## 📂 Dataset Requirements

The CSV file (`full_sensor_data.csv`) must contain the following
columns:

-   `temperature`
-   `humidity`
-   `mq2_value`

Example:

temperature,humidity,mq2_value 36.8,62,780 38.1,58,820 35.5,65,750

------------------------------------------------------------------------

## 🛠 Installation

### 1️⃣ Install Dependencies

pip install numpy pandas scikit-learn joblib

### 2️⃣ Place Dataset

Ensure `full_sensor_data.csv` is in the same directory as the script.

------------------------------------------------------------------------

## 🚀 How to Run

python train_model.py

The script will:

-   Compute efficiency
-   Train the model
-   Print:
    -   R² score
    -   Mean Squared Error
-   Save model as:

biogas_efficiency_model.pkl

------------------------------------------------------------------------

## 📊 Model Details

-   **Algorithm**: Random Forest Regressor
-   **Train/Test Split**: 80/20
-   **Evaluation Metrics**:
    -   R² Score
    -   Mean Squared Error (MSE)
-   **Model Persistence**: `joblib`

------------------------------------------------------------------------

## 💾 Using the Saved Model

``` python
import joblib
import numpy as np

model = joblib.load('biogas_efficiency_model.pkl')

sample = np.array([[37.5, 60, 800]])
prediction = model.predict(sample)

print("Predicted Efficiency:", prediction[0])
```

------------------------------------------------------------------------

## 🔬 Use Case

This model is suitable for:

-   🏭 Biogas plant monitoring systems\
-   📡 IoT-based biogas efficiency tracking\
-   🤖 ESP32/Arduino sensor integration projects\
-   📈 Real-time efficiency prediction dashboards

------------------------------------------------------------------------

## ⚠ Notes

-   The efficiency formula is domain-assumed and based on typical
    anaerobic digestion ranges.
-   For real industrial deployment, calibration with actual plant data
    is recommended.
-   MQ2 values are treated as methane indicators (ensure calibration for
    real-world accuracy).

------------------------------------------------------------------------

## 📌 Future Improvements

-   Add real methane % calibration
-   Deploy as Flask/FastAPI API
-   Integrate with ESP32 live data
-   Hyperparameter tuning
-   Add visualization dashboard

------------------------------------------------------------------------

## 👨‍💻 Author

Developed for biogas efficiency analysis using machine learning.
