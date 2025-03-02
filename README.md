# DeltaPred
 Interactive Model Prediction & Visualization

 🚧 **This project is currently in the prototype phase.**  
Expect frequent updates, and features may change or be incomplete.

## Introduction  
DeltaPred is an interactive model prediction and visualization tool built with **Streamlit** and **Plotly**.

This project was initiated based on experience with structured data competitions, particularly in **regression** and **time-series forecasting**. In many cases, when working with domain-specific datasets, predictions may deviate unexpectedly, requiring in-depth analysis to determine the cause.

For example, in fields like **manufacturing process data**, where the dataset captures the flow of operations, domain expertise can be leveraged to test **how modifying certain input variables impacts the target outcome**. By allowing users to interactively adjust feature values and observe prediction changes, DeltaPred serves as a valuable tool for **debugging models and gaining deeper insights into data-driven decision-making**.

With DeltaPred, users can:
- Upload pre-trained models (**XGBoost, PyTorch, Scikit-learn**)
- Upload datasets (**CSV format**)
- Visualize actual vs. predicted values
- Modify feature values interactively to observe changes in predictions
- Export modified predictions for further analysis


## Installation & Usage

### Run with Docker
1. Build the Docker Image
```bash
docker build -t deltapred .
```
2. Run the Container
```bash
docker run -p 8501:8501 deltapred
```  
Now Open your browser and go to ```http://localhost:8501/```.

### Run Locally
1. Clone the Repository
```bash
git clone https://github.com/minnnnnnnn-dev/DeltaPred.git
cd DeltaPred
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Run Streamlut App
```bash
streamlit run dashboard/app.py
```  
The application will be accessible at ```http://localhost:8501/```.


## Current Features
- Upload trained models (**XGBoost, PyTorch, Scikit-learn**)  
- Upload tabular datasets (**CSV format**)  
- Feature selection for prediction input  
- Graph visualization using **Plotly**  
- Modify feature values to analyze prediction changes  
- Zoom functionality to focus on specific data points  
- Docker containerization for easy deployment  


## Upcoming Features
- Support for **time-series forecasting**  
- Implement **automated feature importance analysis**  
- Add **model explanation with SHAP values**  
- Support additional model frameworks (**TensorFlow, LightGBM, CatBoost**)  
- Enhance **UI/UX with interactive widgets**  
- Provide an **API endpoint for external model inference**  