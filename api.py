import base64
import json
import time
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# Import your project modules.
# These modules must be accessible within your project's structure.
from dataset import rate_code  # function that converts raw features to spike trains
from snn import FraudSNN
from snn_hfe import (
    get_quantize_module,
    get_quantize_input,
    get_encrypted_input,
    get_encrypted_output,
    get_decrypted_output,
    get_dequantize_output,
    get_normal_output,
)

# ----------------------------
# Load configuration from config.json
# ----------------------------
try:
    with open("config.json", "r") as file:
        Config = json.load(file)
except Exception as e:
    raise RuntimeError("Error loading config.json: " + str(e))

# ----------------------------
# Initialize FastAPI app
# ----------------------------
app = FastAPI(
    title="SNN FHE Prediction API",
    description="An API endpoint that receives feature values from the frontend and processes them through the FHE pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# ----------------------------
# Model and Pipeline Initialization
# ----------------------------
device = torch.device("cpu")  # Using CPU for FHE compilation

# Set model parameters using the configuration.
input_size = Config.get("input_size", 11)  # Default to 11 features if not specified
hidden_size = Config["hidden_size"]
time_steps = Config["time_steps"]  # For instance, 2 for the unrolled version

# Initialize the FraudSNN model.
snn_classifier = FraudSNN(input_size, hidden_size, time_steps, beta=0.9, threshold=1.0)

# Load the pre-trained model weights.
try:
    snn_classifier.load_state_dict(
        torch.load("saved_models/snn_model.pth", map_location=device)
    )
    print("Model loaded from saved_models/snn_model.pth")
except FileNotFoundError:
    raise RuntimeError("snn_model.pth not found. Please train the model first.")

snn_classifier.eval()  # Set the model to evaluation mode

# Pre-compile the model for FHE inference using a dummy input.
dummy_input = torch.randn((1, time_steps, input_size), dtype=torch.float32)
try:
    quantized_module, _ = get_quantize_module(
        snn_classifier, dummy_input, method="approximate", bits=8
    )
    print("Model compiled for FHE inference.")
except Exception as e:
    raise RuntimeError("Error compiling model for FHE inference: " + str(e))


# ----------------------------
# Define the Pydantic Model for Request Validation
# ----------------------------
class PredictionInput(BaseModel):
    # This field expects a list of numerical values representing features.
    features: list[float] # 11 numbers for the 11 features
    # Example: [0.1, 0.2, 0.3, ..., 0.11] 


# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict/")
async def predict(input_data: PredictionInput):
    try:
        print('hello', input_data)
        # Retrieve feature list from the JSON sent by the frontend.
        features_array = np.array(input_data.features)

        # Transform the raw features into a spike train.
        # The rate_code function should return an array with shape [time_steps, features].
        spikes = rate_code(features_array)

        # Convert the spike train to a torch tensor and add a batch dimension.
        torch_input = torch.tensor(spikes, dtype=torch.float32).unsqueeze(0)

        # ----------------------------
        # FHE Inference Pipeline
        # ----------------------------

        # 1. Quantize the input data.
        q_input, _ = get_quantize_input(quantized_module, torch_input)

        # 2. Encrypt the quantized input.
        q_input_enc, _ = get_encrypted_input(quantized_module, q_input)
        encrypted_input_str = base64.b64encode(q_input_enc.serialize()).decode("utf-8")

        # 3. Execute the encrypted inference.
        q_y_enc, _ = get_encrypted_output(quantized_module, q_input_enc)
        encrypted_output_str = base64.b64encode(q_y_enc.serialize()).decode("utf-8")

        # 4. Decrypt the output.
        q_y, _ = get_decrypted_output(quantized_module, q_y_enc)

        # 5. De-quantize and post-process the output.
        dequantized_output, _ = get_dequantize_output(quantized_module, q_y)

        # Also obtain the output from the normal (unencrypted) model for comparison.
        normal_output, _ = get_normal_output(snn_classifier, torch_input)
        
  

        # Prepare the response dictionary.
        response = {
            "encrypted_input": encrypted_input_str,
            "encrypted_output": encrypted_output_str,
            "dequantized_output": (
                dequantized_output.tolist()
                if isinstance(dequantized_output, (torch.Tensor, np.ndarray))
                else dequantized_output
            ),
            "normal_output": (
                normal_output.tolist()
                if isinstance(normal_output, (torch.Tensor, np.ndarray))
                else normal_output
            ),
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(e))


# ----------------------------
# To run the API, use the command:
#   uvicorn app:app --reload
# ----------------------------
