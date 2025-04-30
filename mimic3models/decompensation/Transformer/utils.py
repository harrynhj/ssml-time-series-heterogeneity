
# from tensorflow.python.keras.utils.data_utils import Sequence

import common_utils
import threading
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence


# --- Preprocess Chunk Function (Unchanged) ---
def preprocess_chunk(data, ts, discretizer, normalizer=None):
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    return data

# --- BatchGen Class (Original, to be used with tf.data.Dataset.from_generator) ---
# No changes needed inside this class definition itself to use it with from_generator.
# We will create the Dataset object in the main script.
class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer,
                 batch_size, steps, shuffle, return_names=False):
        self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names # Note: return_names=True needs special handling with model.fit

        if steps is None:
            self.n_examples = reader.get_number_of_examples()
            self.steps = (self.n_examples + batch_size - 1) // batch_size
        else:
            self.n_examples = steps * batch_size
            self.steps = steps

        self.chunk_size = min(1024, self.steps) * batch_size
        self.lock = threading.Lock() # Lock might cause issues with tf.data, monitor performance
        self.generator = self._generator()

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                self.reader.random_shuffle()
            remaining = self.n_examples
            # print("Starting new epoch (or first epoch). Total examples: {}".format(remaining)) # Debug print
            current_epoch_step = 0
            while remaining > 0:
                current_size = min(self.chunk_size, remaining)
                remaining -= current_size

                # print("Reading chunk, size: {}".format(current_size)) # Debug print
                ret = common_utils.read_chunk(self.reader, current_size)
                Xs = ret["X"]
                ts = ret["t"]
                ys = ret["y"]
                names = ret["name"]

                # print("Preprocessing chunk...") # Debug print
                Xs = preprocess_chunk(Xs, ts, self.discretizer, self.normalizer)
                (Xs, ys, ts, names) = common_utils.sort_and_shuffle([Xs, ys, ts, names], B)
                # print("Chunk processed and shuffled.") # Debug print

                for i in range(0, current_size, B):
                    # print("Yielding batch {}/{}".format(current_epoch_step + 1, self.steps)) # Debug print
                    X = common_utils.pad_zeros(Xs[i:i + B])
                    y = np.array(ys[i:i + B])
                    batch_names = names[i:i+B]
                    batch_ts = ts[i:i+B]
                    batch_data = (X, y)

                    # IMPORTANT: model.fit expects (x, y) or (x, y, sample_weight)
                    # It cannot directly consume the dictionary structure.
                    # If return_names is True, this generator cannot be directly used
                    # by model.fit for training. It might be used for evaluation separately.
                    if self.return_names:
                         # Yielding dict for separate evaluation use?
                         # Or raise error if used for training?
                         # For now, let's assume if used for fit, return_names must be False
                         # Or handle it in the from_generator call if possible/needed
                         # For simplicity now, assume yield for fit needs (X, y)
                         # raise ValueError("BatchGen with return_names=True cannot be directly used for model.fit training")
                         # If needing names during training (e.g. custom loss), need different approach
                         yield {"data": batch_data, "names": batch_names, "ts": batch_ts} # Keep original for now, handle usage in main.py
                    else:
                        yield batch_data
                    current_epoch_step += 1
            # print("Epoch finished.") # Debug print


    # Make the class iterable
    def __iter__(self):
        # Reset or ensure the generator starts fresh if needed,
        # depending on how _generator is implemented. Here it's infinite.
        # For from_generator, returning self.generator might be enough if it yields correctly.
        # Let's stick to returning the generator instance.
        return self.generator

    # next() methods are often not needed when using __iter__ with from_generator
    # but keeping them doesn't hurt if they were used elsewhere.
    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()


# --- BatchGenDeepSupervision Class (Refactored to inherit from keras.utils.Sequence) ---
class BatchGenDeepSupervision(Sequence): # Inherit from Sequence

    def __init__(self, dataloader, discretizer, normalizer,
                 batch_size, shuffle, return_names=False):
        # Store parameters
        self.dataloader = dataloader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        # IMPORTANT: Sequence __getitem__ must return (x, y) or (x, y, sample_weight) for model.fit
        # Storing return_names but __getitem__ won't use it directly for fit.
        self.return_names_flag = return_names
        if self.return_names_flag:
            print("Warning: BatchGenDeepSupervision created with return_names=True. "
                  "The __getitem__ method used by model.fit will return only (x, y). "
                  "Names/timestamps need separate handling if required during training/evaluation.")

        # Load and preprocess all data into memory
        self._load_per_patient_data()

        # Calculate steps per epoch
        self.n_samples = len(self.data[1]) # Number of samples (patients)
        self.steps = (self.n_samples + self.batch_size - 1) // self.batch_size

        # Initialize indices and shuffle if needed
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_per_patient_data(self):
        # Renamed original args to self.xxx
        dataloader = self.dataloader
        discretizer = self.discretizer
        normalizer = self.normalizer
        timestep = discretizer._timestep

        def get_bin(t):
            eps = 1e-6
            return int(t / timestep - eps)

        N = len(dataloader._data["X"])
        Xs = []
        ts_original = [] # Keep original timestamps if needed for return_names=True later
        masks = []
        ys = []
        names = []

        for i in range(N):
            X = dataloader._data["X"][i]
            cur_ts = dataloader._data["ts"][i]
            cur_ys = dataloader._data["ys"][i]
            name = dataloader._data["name"][i]

            cur_ys = [int(x) for x in cur_ys]

            T = 0 if len(cur_ts) == 0 else max(cur_ts) # Handle empty case
            nsteps = get_bin(T) + 1
            mask = np.zeros(nsteps, dtype=int) # Use numpy array directly
            y = np.zeros(nsteps, dtype=int)    # Use numpy array directly

            for pos, z in zip(cur_ts, cur_ys):
                bin_index = get_bin(pos)
                if 0 <= bin_index < nsteps: # Ensure index is within bounds
                    mask[bin_index] = 1
                    y[bin_index] = z

            # Pass T to transform, ensure T is not negative if cur_ts was empty
            X, header = discretizer.transform(X, end=max(0, T))
            if normalizer is not None:
                X = normalizer.transform(X)

            # Ensure X shape matches mask/y length after transform
            if X.shape[0] != nsteps:
                 # This might happen due to discretization details or empty inputs
                 # Option 1: Pad/truncate X (potentially problematic)
                 # Option 2: Adjust mask/y based on X's actual length (safer if transform dictates length)
                 # Let's assume transform dictates length for now, adjust mask/y
                 print(f"Warning: Length mismatch for sample {name}. X shape {X.shape[0]}, nsteps {nsteps}. Adjusting mask/y.")
                 nsteps = X.shape[0]
                 mask = mask[:nsteps] if len(mask) > nsteps else np.pad(mask, (0, nsteps - len(mask)))
                 y = y[:nsteps] if len(y) > nsteps else np.pad(y, (0, nsteps - len(y)))


            Xs.append(X) # X is already a numpy array
            masks.append(mask) # mask is numpy array
            ys.append(y)       # y is numpy array
            names.append(name)
            ts_original.append(cur_ts) # Store original ts

            # Assertions might need adjustment based on how empty sequences are handled
            # assert np.sum(mask) > 0 or T == 0 # Allow empty sequences
            assert len(X) == len(mask) and len(X) == len(y)

        # Store data needed for __getitem__
        # self.data format: [[List of X arrays, List of mask arrays], List of y arrays]
        self.data = [[Xs, masks], ys]
        # Store names and original ts separately for potential later use if return_names_flag is True
        self.patient_names = names
        self.patient_ts = ts_original

    def __len__(self):
        # Return the number of batches per epoch
        return self.steps

    def __getitem__(self, index):
        # Generate indices of the batch
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start_index:end_index]

        # Get data for the batch
        batch_X = [self.data[0][0][k] for k in batch_indices]
        batch_mask = [self.data[0][1][k] for k in batch_indices]
        batch_y = [self.data[1][k] for k in batch_indices]

        # Pad data for the batch
        padded_X = common_utils.pad_zeros(batch_X)      # (B, T, D)
        padded_mask = common_utils.pad_zeros(batch_mask)  # (B, T)
        padded_y = common_utils.pad_zeros(batch_y)
        padded_y = np.expand_dims(padded_y, axis=-1)    # (B, T, 1)

        # model.fit expects (x, y) or (x, y, sample_weight)
        # x can be a list/tuple if the model has multiple inputs
        # Here x is [padded_X, padded_mask], y is padded_y
        return ([padded_X, padded_mask], padded_y)

    def on_epoch_end(self):
        # Shuffle indices after each epoch if shuffle is True
        if self.shuffle:
            np.random.shuffle(self.indices)

    # --- Methods below are no longer needed for Sequence interface ---
    # def _generator(self):
    #     ...
    # def __iter__(self):
    #     ...
    # def next(self):
    #     ...
    # def __next__(self):
    #     ...

# --- save_results function (Unchanged) ---
def save_results(names, ts, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,period_length,prediction,y_true\n")
        for (name, t, x, y) in zip(names, ts, pred, y_true):
            # Ensure name is treated as string, handle potential commas within name if needed
            f.write("{},{:.6f},{:.6f},{}\n".format(f'"{name}"' if ',' in str(name) else name, t, x, y))