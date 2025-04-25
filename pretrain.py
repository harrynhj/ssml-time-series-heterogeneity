# pretrain_base_lstm.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import datetime
from tqdm import tqdm
# Assuming your custom modules are importable (e.g., in the same directory or added to PYTHONPATH)
from rnnmodel import LSTM #
import data #
# Import necessary dependencies for data loading, if not handled within data.py already
# import preprocessing
# import reader
# import utils

# --- Hyperparameters and Configuration ---
DATASET_PATH = './data/decompensation' # !! IMPORTANT: Set your actual data path !!
SAVE_PATH = './pretrain/onelayer128.pth' # !! IMPORTANT: Set where you want to save the pre-trained model !!
LEARNING_RATE = 2e-3
BATCH_SIZE = 8 # Match this with data loading if possible, but the loop below processes patient by patient
EPOCHS = 50 # Adjust based on convergence
DROPOUT_RATE = 0.3 # Set your desired dropout rate
NUM_BATCHES_PER_EPOCH = 100 # Adjust based on dataset size and desired epoch length for data loading

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure save directory exists
save_dir = os.path.dirname(SAVE_PATH)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

# --- Data Loading ---
# load_all_data returns a list of tuples: (list_of_sequences, label, label_true)
# Note: The original 'data.py' loads data suitable for the meta-learning setup.
# The BatchGen inside seems tailored for that structure.
# We might need to iterate differently here for standard training.
# This example assumes we process patient by patient from the loaded data.
print("Loading data...")
# We get data_all (list of patient tuples) and data_aug (list of augmented sequences)
# For pre-training, we likely only need data_all. data_aug seems specific to the meta-learning task.
data_all, _ = data.load_all_data(DATASET_PATH, BATCH_SIZE, NUM_BATCHES_PER_EPOCH, mode='train')
# TODO: Optionally load validation data similarly if you want validation loops
# val_data_all, _ = data.load_all_data(DATASET_PATH, BATCH_SIZE, NUM_BATCHES_PER_EPOCH // 5, mode='val')
print(f"Loaded {len(data_all)} patient records for training.")


# --- Model, Loss, Optimizer ---
# Model parameters based on train.py and rnnmodel.py
d_input = 76
d_hidden = 128
d_output = 1 # Single output for binary classification

# Instantiate the LSTM model
# Note: The 'batch' argument in LSTM init seems unused in its forward pass, using BATCH_SIZE for consistency.
model = LSTM(input_size=d_input, hidden_size=d_hidden, output_size=d_output, dropout=DROPOUT_RATE, batch=BATCH_SIZE).to(device)

# Loss Function: Binary Cross Entropy with Logits (handles sigmoid internally)
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    num_samples = 0

    # Iterate through patient data provided by load_all_data
    # Each 'patient_data' is a tuple: (list_of_sequences, label, label_true)
    for patient_sequences, patient_label, _ in tqdm(data_all, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        if not patient_sequences: # Skip if a patient has no sequences
            continue

        # --- Data Handling Strategy ---
        # Decide how to use the sequences for prediction. Common strategies:
        # 1. Use only the *last* sequence segment for prediction.
        # 2. Run LSTM over *all* segments and use the final hidden state.
        # Strategy 1 (Simpler): Use the last sequence segment
        last_sequence = patient_sequences[-1] # Shape: (24, 76)

        # Prepare input and target tensors
        # Input needs batch dimension: [1, seq_len, features] -> [1, 24, 76]
        input_tensor = torch.Tensor(last_sequence).unsqueeze(0).to(device)
        # Target label needs to be a tensor, shape [1] for BCEWithLogitsLoss
        #print(f"DEBUG: patient_label = {patient_label}, type = {type(patient_label)}")
        scalar_label = patient_label.item()  # Convert numpy array (e.g., array(0)) to scalar (e.g., 0)
        target_tensor = torch.tensor([scalar_label], dtype=torch.float32).to(device)  # Use the scalar value

        # --- Forward Pass ---
        optimizer.zero_grad()
        # The LSTM model expects batch dimension first, output shape is [batch_size]

        output = model(input_tensor) # Output shape should be [1]

        # --- Loss & Backward Pass ---
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_samples += 1

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    print(f"Epoch {epoch+1}/{EPOCHS} - Average Training Loss: {avg_loss:.4f}")

    # --- Optional: Validation Step ---
    # if (epoch + 1) % 1 == 0: # Validate every epoch (or less frequently)
    #     model.eval()
    #     with torch.no_grad():
    #         # Create a similar loop for validation data
    #         # Calculate validation loss and other metrics (AUC, etc.)
    #         pass # Placeholder for validation logic
    #     # TODO: Add logic here to save the model if validation performance improves

# --- Save Final Model ---
print(f"Training finished. Saving model to {SAVE_PATH}")
torch.save(model.state_dict(), SAVE_PATH)
print("Model saved successfully.")