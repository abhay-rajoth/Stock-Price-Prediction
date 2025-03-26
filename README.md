# 📈 Forex Price Prediction using LSTM  

## 🔹 Overview  
This project builds a **Long Short-Term Memory (LSTM) model** to predict **forex currency prices** using historical market data.  
The model leverages **time-series forecasting techniques, feature engineering, and deep learning** to identify patterns in exchange rates and forecast future prices.  

## 🎯 Key Features  
✅ Preprocesses and normalizes forex data for better model performance.  
✅ Implements **technical indicators** like Moving Averages (MA), RSI, and MACD.  
✅ Uses **LSTM neural networks** to capture sequential dependencies in price movements.  
✅ Evaluates performance using **MSE & R² Score** to ensure accuracy.  
✅ Visualizes actual vs predicted prices using **Matplotlib & Power BI dashboards**.  

---

## 📂 Dataset  
The dataset consists of **historical forex price data** for major currency pairs (e.g., **EUR/USD, GBP/USD**). It includes:  

| Column      | Description                              |
|------------|------------------------------------------|
| Date       | Timestamp of the forex record           |
| Open       | Opening price of the currency pair      |
| High       | Highest price reached in the period     |
| Low        | Lowest price reached in the period      |
| Close      | Closing price of the currency pair      |
| Volume     | Number of trades executed               |

🔹 **Source:** [Update with actual data source if available]  

---

## 🛠️ Installation & Setup  

### 🔹 Prerequisites  
Ensure you have the following installed:  
- Python 3.8+  
- Jupyter Notebook / VS Code  
- TensorFlow/Keras  
- Pandas, NumPy, Matplotlib  
- Scikit-learn  

### 🔹 Install Dependencies  
Run the following command to install required libraries:  
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```
## 🔍 Methodology  

### **1️⃣ Data Preprocessing**  
- Converted **Date** column to datetime format.  
- Normalized price data using **MinMaxScaler**.  
- Created **lag features** and **moving averages** for trend analysis.  

### **2️⃣ Feature Engineering**  
- **Lag Features:** `Close_lag1`, `Close_lag7`, `Close_lag30`.  
- **Technical Indicators:**  
  - **Moving Averages (MA_7, MA_30)**  
  - **Relative Strength Index (RSI)**  
  - **MACD Indicator**  

### **3️⃣ Model Architecture**  
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
```
### **4️⃣ Model Training**
-Used Adam & RMSprop optimizers for better convergence.
-Trained for 50+ epochs with batch size 16.
-Evaluated using **Mean Squared Error (MSE)** and **R² Score**.

#📊 Results & Performance
##📈 Actual vs Predicted Prices
```
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Prices", color='blue')
plt.plot(y_pred, label="Predicted Prices", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Forex Price")
plt.title("LSTM Forex Price Prediction: Actual vs Predicted")
plt.legend()
plt.grid()
plt.show()
```
**📌 Insight: The model successfully captured upward and downward trends in forex prices.**
##📉 Loss Curve
```
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()
plt.show()
```
