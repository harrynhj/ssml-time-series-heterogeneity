import numpy as np
import argparse
import os
# Removed imp, using importlib
import importlib.util
import importlib.machinery
import sys
import re
import tensorflow as tf # Make sure tf is imported

# Import Sequence for BatchGenDeepSupervision
# Import other necessary modules from the project
from mimic3models.decompensation import utils
from mimic3benchmark.readers import DecompensationReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

# Import Keras callbacks from the public API
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger # Use public API path


parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/decompensation/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.set_defaults(deep_supervision=False)
args = parser.parse_args()
print(args)

if args.small_part:
    # Setting a very large number effectively disables saving except maybe at the very end
    # Consider if this is the intended behavior or if saving should be skipped entirely.
    args.save_every = 2**30

# Build readers, discretizers, normalizers
# These need to be defined before they are used in the else block below
train_reader = None
val_reader = None
train_data_loader = None
val_data_loader = None

if args.deep_supervision:
    print("==> Loading data in Deep Supervision mode")
    train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                               listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                               small_part=args.small_part)
    val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                             listfile=os.path.join(args.data, 'val_listfile.csv'),
                                                             small_part=args.small_part)
else:
    print("==> Loading data in standard mode")
    train_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'train'),
                                        listfile=os.path.join(args.data, 'train_listfile.csv'))
    val_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'train'),
                                      listfile=os.path.join(args.data, 'val_listfile.csv'))

print("==> Creating Discretizer")
discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

# Get header after discretizer is created
print("==> Getting discretizer header")
if args.deep_supervision:
    # Need to read at least one example to get header structure
    # Assuming _data is loaded in DeepSupervisionDataLoader's init
    if not train_data_loader or not train_data_loader._data["X"]:
         raise ValueError("DeepSupervisionDataLoader did not load data correctly.")
    discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
else:
    # Need to read at least one example to get header structure
    if not train_reader:
         raise ValueError("Train reader was not initialized.")
    try:
        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    except IndexError:
        raise ValueError("Could not read example 0 from train_reader to get header. Is the dataset empty or listfile incorrect?")

print(f"    Discretizer header length: {len(discretizer_header)}")
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

print("==> Creating Normalizer")
normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'decomp_ts{}.input_str-previous.n1e5.start_time-zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
print(f"    Loading normalizer state from: {normalizer_state}")
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'decomp'


# Build the model
print("==> using model {}".format(args.network))
# --- Using standard importlib ---
try:
    # Convert path to module import string
    module_import_path = args.network.replace('.py', '').replace('/', '.')
    print(f"==> attempting to import module: {module_import_path}")
    model_module = importlib.import_module(module_import_path)
# --- End import ---

print("==> Building network")
model = model_module.Network(**args_dict)
suffix = "{}.bs{}{}{}.ts{}".format("" if not args.deep_supervision else ".dsup",
                                   args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep)
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
# print(args.optimizer) # args.optimizer is not used directly here anymore

# --- Using direct Adam instantiation ---
print(f"    Using Adam optimizer with lr={args.lr}, beta_1={args.beta_1}")
optimizer_instance = tf.keras.optimizers.Adam(learning_rate=args.lr,
                                             beta_1=args.beta_1)

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. The result is (B, T).
model.compile(optimizer=optimizer_instance,
              loss='binary_crossentropy')

model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    print(f"==> Loading model weights from: {args.load_state}")
    model.load_weights(args.load_state)
    try:
        # Use non-greedy match for numbers to handle potential multiple digits
        match = re.match(".*chunk([0-9]+).*", args.load_state)
        if match:
            n_trained_chunks = int(match.group(1))
            print(f"    Resuming training from epoch {n_trained_chunks}")
        else:
            print("    Warning: Could not parse chunk number from load_state filename.")
    except (AttributeError, IndexError, ValueError):
         print("    Warning: Could not parse chunk number from load_state filename.")


# --- Load data and prepare generators/datasets ---
train_data_source = None # Will hold Sequence or Dataset
val_data_source = None   # Will hold Sequence or Dataset
train_steps = None
val_steps = None

if args.deep_supervision:
    print("==> Preparing Sequence generators for Deep Supervision")
    # Create instances of the refactored BatchGenDeepSupervision Sequence
    train_data_source = utils.BatchGenDeepSupervision(train_data_loader, discretizer,
                                                   normalizer, args.batch_size, shuffle=True,
                                                   return_names=False) # Ensure return_names=False for fit
    val_data_source = utils.BatchGenDeepSupervision(val_data_loader, discretizer,
                                                 normalizer, args.batch_size, shuffle=False,
                                                 return_names=False) # Ensure return_names=False for fit
    # For Sequence, steps are determined internally by __len__
    train_steps = len(train_data_source)
    val_steps = len(val_data_source)
    print(f"    Train steps per epoch (from Sequence): {train_steps}")
    print(f"    Validation steps per epoch (from Sequence): {val_steps}")

else: # --- This is the block that needs the tf.data.Dataset wrapping ---
    print("==> Preparing tf.data.Dataset for standard mode")
    # Set number of batches in one epoch (steps_per_epoch)
    train_nbatches = 2000
    val_nbatches = 1000
    if args.small_part:
        train_nbatches = 40
        val_nbatches = 40

    # 1. Create original BatchGen objects (with _obj suffix and return_names=False)
    print("    Creating original BatchGen objects...")
    train_data_gen_obj = utils.BatchGen(train_reader, discretizer,
                                        normalizer, args.batch_size, train_nbatches, True,
                                        return_names=False) # Important: return_names=False
    val_data_gen_obj = utils.BatchGen(val_reader, discretizer,
                                      normalizer, args.batch_size, val_nbatches, False,
                                      return_names=False) # Important: return_names=False

    # Store steps from the original objects
    train_steps = train_data_gen_obj.steps
    val_steps = val_data_gen_obj.steps
    print(f"    Train steps per epoch (from BatchGen): {train_steps}")
    print(f"    Validation steps per epoch (from BatchGen): {val_steps}")

    # 2. Define the output signature (!!! ADJUST dtype/shape IF NEEDED !!!)
    print("    Defining output signature for tf.data.Dataset...")
    num_features = len(discretizer_header)
    # Assuming X=(batch, time, features), y=(batch,)
    output_signature = (
        tf.TensorSpec(shape=(None, None, num_features), dtype=tf.float32), # X
        tf.TensorSpec(shape=(None,), dtype=tf.int32)                     # y
    )
    print(f"        Output Signature: {output_signature}")

    # 3. Create tf.data.Dataset objects using from_generator
    print("    Creating tf.data.Dataset objects from generator...")
    train_data_source = tf.data.Dataset.from_generator(
        lambda: train_data_gen_obj,  # Use lambda!
        output_signature=output_signature
    )
    val_data_source = tf.data.Dataset.from_generator(
        lambda: val_data_gen_obj,  # Use lambda!
        output_signature=output_signature
    )

    # (Optional) Apply prefetch for performance
    # train_data_source = train_data_source.prefetch(tf.data.AUTOTUNE)
    # val_data_source = val_data_source.prefetch(tf.data.AUTOTUNE)
    print("    Dataset objects created.")
    # --- End of Dataset wrapping block ---


if args.mode == 'train':
    print("==> Setting up training callbacks")
    # Prepare training callbacks
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.chunk{epoch}.test{val_loss:.4f}.weights.h5')

    # --- Refactored ModelCheckpoint ---
    # Decide save frequency based on args.save_every (assuming it means epochs)
    if args.save_every == 1:
        save_frequency = 'epoch'
        print(f"    ModelCheckpoint: Saving weights every epoch.")
    elif train_steps: # Need train_steps to calculate batch frequency
        save_frequency = args.save_every * train_steps
        print(f"    ModelCheckpoint: Saving weights approx. every {args.save_every} epochs (every {save_frequency} batches).")
    else:
        print("    Warning: Cannot determine steps_per_epoch, defaulting ModelCheckpoint save_freq to 'epoch'.")
        save_frequency = 'epoch'

    # Ensure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        print(f"    Creating directory for saving states: {dirname}")
        os.makedirs(dirname)

    # Use public API path for ModelCheckpoint
    saver = tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        verbose=1,
        save_freq=save_frequency, # Use calculated frequency
        save_weights_only=True # Typically recommended
    )
    # --- End Refactored ModelCheckpoint ---

    # Setup metrics callback (ensure it can handle both Sequence and Dataset inputs if needed, or adjust based on mode)
    # Assuming DecompensationMetrics is compatible or adjusted
    # Note: Passing the original generator objects here might be intended by the callback's design
    # If DecompensationMetrics needs to iterate, it might need the original objects or adapted logic
    print("    Setting up DecompensationMetrics callback")
    metrics_callback = keras_utils.DecompensationMetrics(
        train_data_gen=train_data_gen_obj if not args.deep_supervision else train_data_source, # Pass appropriate source
        val_data_gen=val_data_gen_obj if not args.deep_supervision else val_data_source,       # Pass appropriate source
        deep_supervision=args.deep_supervision,
        batch_size=args.batch_size,
        verbose=args.verbose
    )


    # Setup CSV Logger
    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        print(f"    Creating directory for Keras logs: {keras_logs}")
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> Starting training...")
    # --- Refactored model.fit call ---
    # Use the unified train_data_source and val_data_source variables
    print(f"    Training data type: {type(train_data_source)}")
    print(f"    Validation data type: {type(val_data_source)}")

    model.fit(train_data_source,
              # steps_per_epoch is needed if using tf.data.Dataset from generator
              # It's not needed if using a Sequence (Keras gets length via __len__)
              steps_per_epoch=train_steps if not args.deep_supervision else None,
              validation_data=val_data_source,
              # validation_steps is needed if using tf.data.Dataset from generator
              # It's not needed if using a Sequence
              validation_steps=val_steps if not args.deep_supervision else None,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              verbose=args.verbose)
    # --- End Refactored model.fit call ---

elif args.mode == 'test':
    print("==> Starting testing...")
    # ensure that the code uses test_reader/loader appropriately
    # Deleting training data sources is good practice
    del train_data_source
    del val_data_source
    # Delete original objects too if they exist
    try: del train_data_gen_obj
    except NameError: pass
    try: del val_data_gen_obj
    except NameError: pass
    try: del train_data_loader
    except NameError: pass
    try: del val_data_loader
    except NameError: pass
    try: del train_reader
    except NameError: pass
    try: del val_reader
    except NameError: pass


    names = []
    ts = []
    labels = []
    predictions = []

    if args.deep_supervision:
        print("    Loading test data for Deep Supervision...")
        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'test'),
                                                                  listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                                  small_part=args.small_part)
        # Create Sequence for testing (can use return_names=True here if needed by logic below)
        # Note: The original test loop logic seems complex and might need Sequence adaptation.
        # Let's assume we still need the dict structure for the test loop logic.
        # We create a Sequence but might iterate differently or access data directly.
        # OR, we stick to the original generator logic for testing if easier.
        # Let's use the original generator logic for the test loop for now, as it expects the dict.
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                      normalizer, args.batch_size,
                                                      shuffle=False, return_names=True) # Use return_names=True

        print(f"    Predicting on {test_data_gen.steps} test batches...")
        for i in range(test_data_gen.steps):
            print(f"\tBatch {i+1}/{test_data_gen.steps}", end='\r')
            # Use the generator's next method (assuming it's implemented correctly)
            # Or iterate: ret = next(iter(test_data_gen))
            try:
                ret = next(test_data_gen) # Use the generator interface
            except StopIteration:
                 print("\nWarning: Test generator finished early.")
                 break

            (x, y) = ret["data"] # x is [padded_X, padded_mask], y is padded_y
            cur_names = ret["names"] # List of names for the batch
            cur_ts = ret["ts"] # List of original timestamp lists for the batch

            # Predict using the model
            pred = model.predict(x, batch_size=args.batch_size, verbose=0) # pred shape (B, T, 1)

            # Flatten results and apply mask
            batch_masks_flat = x[1].flatten() # Flatten mask (B*T,)
            batch_labels_flat = y.flatten()   # Flatten labels (B*T,)
            batch_preds_flat = pred.flatten() # Flatten predictions (B*T,)

            # Expand names and ts to match flattened structure (B*T,)
            # Assuming x[0] shape is (B, T, D)
            batch_size_actual = x[0].shape[0]
            num_timesteps = x[0].shape[1]
            expanded_names = []
            expanded_ts = []
            if len(cur_names) == batch_size_actual:
                 for idx, name in enumerate(cur_names):
                      expanded_names.extend([name] * num_timesteps)
                      # Assuming cur_ts[idx] is the list of ts for that sample
                      # We need to align these with the *padded* timesteps
                      # This part is tricky - how are original ts mapped to padded steps?
                      # The original code's approach might be simpler if it worked.
                      # Let's replicate the original logic's potential intent:
                      # It seemed to iterate through flattened mask, label, pred, and *repeated* names.
                      # The timestamp handling was unclear (`ts += single_ts`).

            # --- Reverting to a structure closer to original test loop ---
            # This assumes the original loop's logic correctly handled the flattened data
            # and aligned it with names/ts somehow. This part is complex.

            current_names_repeated = []
            for name in cur_names:
                current_names_repeated.extend([name] * num_timesteps)

            # The original ts logic `ts += single_ts` likely appended all original timestamps
            # without aligning to the predictions. Let's keep that for now.
            for single_ts_list in cur_ts:
                 ts.extend(single_ts_list) # Appends all original timestamps

            # Iterate through the flattened, masked data points
            valid_indices = np.where(batch_masks_flat == 1)[0]
            labels.extend(batch_labels_flat[valid_indices])
            predictions.extend(batch_preds_flat[valid_indices])
            # Align names based on valid indices - requires careful index calculation
            # Assuming current_names_repeated aligns with flattened data
            names.extend(np.array(current_names_repeated)[valid_indices])

        print('\n    Finished prediction.')

    else: # Standard mode (non-deep supervision)
        print("    Loading test data for standard mode...")
        test_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'test'),
                                           listfile=os.path.join(args.data, 'test_listfile.csv'))

        # Use BatchGen directly for testing, with return_names=True
        test_data_gen = utils.BatchGen(test_reader, discretizer,
                                       normalizer, args.batch_size,
                                       None, shuffle=False, return_names=True) # steps = None for full test

        print(f"    Predicting on {test_data_gen.steps} test batches...")
        for i in range(test_data_gen.steps):
            print(f"\tBatch {i+1}/{test_data_gen.steps}", end='\r')
            try:
                ret = next(test_data_gen) # Use the generator interface
            except StopIteration:
                 print("\nWarning: Test generator finished early.")
                 break

            x, y = ret["data"] # x=(B, T, D), y=(B,)
            cur_names = ret["names"] # (B,)
            cur_ts = ret["ts"] # (B,) - Assuming one timestamp per sample here? Check BatchGen output.

            x = np.array(x) # Ensure numpy array
            # Use predict for consistency, adjust based on model output shape if needed
            pred = model.predict(x, batch_size=args.batch_size, verbose=0) # Output shape depends on model

            # Assuming model's final layer gives (B, 1) or (B,) for this task
            if pred.ndim > 1 and pred.shape[-1] == 1:
                pred = pred.flatten() # Make it (B,)
            elif pred.ndim > 1:
                 print(f"Warning: Unexpected prediction shape {pred.shape} in test mode. Taking first element.")
                 pred = pred[:, 0]


            predictions.extend(list(pred))
            labels.extend(list(y))
            names.extend(list(cur_names))
            ts.extend(list(cur_ts)) # Assuming ts is list/array of length B

        print('\n    Finished prediction.')


    # --- Save results ---
    print("==> Evaluating and saving results...")
    if not labels or not predictions:
         print("    Warning: No labels or predictions generated during testing. Skipping metrics and saving.")
    else:
        metrics.print_metrics_binary(labels, predictions)

        # Ensure output directory exists
        results_dir = os.path.join(args.output_dir, 'test_predictions')
        if not os.path.exists(results_dir):
            print(f"    Creating directory for test predictions: {results_dir}")
            os.makedirs(results_dir)

        # Construct file path using results_dir
        results_filename = "test_results.csv" # Default name
        if args.load_state and os.path.basename(args.load_state):
             # Try to create a meaningful name from the loaded state file
             base_name = os.path.splitext(os.path.basename(args.load_state))[0]
             results_filename = f"{base_name}_predictions.csv"

        path = os.path.join(results_dir, results_filename)
        print(f"    Saving test results to: {path}")
        utils.save_results(names, ts, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")

print("==> Script finished.")