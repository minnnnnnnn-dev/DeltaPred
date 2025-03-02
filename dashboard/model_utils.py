import torch
import pickle
import joblib
import xgboost as xgb
import io

def load_model(model_file):
    """
    Load the Uploaded Model
    
    Currently supported model types:
    - PyTorch (`.pt`, `.pth`)
    - Scikit-learn (`.pkl`, `.sav`)
    - XGBoost (`.pkl`)
    """
    
    file_bytes = model_file.read()  
    file_buffer = io.BytesIO(file_bytes) 
    
    # PyTorch
    if model_file.name.endswith('.pt') or model_file.name.endswith('.pth'):
        model = torch.load(file_buffer, map_location=torch.device('cpu'))
        model.eval()
        return model, "pytorch"
    
    
    # elif model_file.name.endswith(".h5") or model_file.name.endswith(".keras"):
    #     # TensorFlow/Keras 
    #     model = tf.keras.models.load_model(model_file)
    #     return model, "tensorflow"
    
    # Pickle Files
    elif model_file.name.endswith(".pkl"):
        model = pickle.load(file_buffer)
        if isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.XGBRegressor):
            return model, "xgboost"
        else:
            return model, "sklearn"
    
    # Joblib Files
    elif model_file.name.endswith(".sav"):
        model = joblib.load(file_buffer)
        return model, "sklearn"
    
    else:
        return None, "unsupported"
    
    
def predict(model, model_type, df, target_col):
    """
    Run Predictions based on the Model Type
    """
    
    if target_col in df.columns:
        df = df.drop(columns=[target_col]).copy()
    
    
    if model_type == "pytorch":
        input_data = df.to_numpy()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            output = model(input_tensor)
            return output.numpy()
    
    # elif model_type == "tensorflow":
    #     output = model.predict(input_data)
    #     return output.tolist()
    
    elif model_type == "xgboost":
        return model.predict(df).tolist()
    
    elif model_type == "sklearn":
        return model.predict(df).tolist()
    
    else:
        return None