import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import json
import random
import sys

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import helper modules
from preprocessing import Discretizer, Normalizer
from reader import InHospitalMortalityReader, DecompensationReader, LengthOfStayReader, PhenotypingReader
from utils import BatchGen
from metrics import print_metrics_binary, print_metrics_multilabel, print_metrics_regression
import common_utils
from networks import create_network

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data(args):
    """
    Load and preprocess data for the selected task.

    Returns:
        train_data_gen: Generator for training data
        val_data_gen: Generator for validation data
        test_data_gen: Generator for test data
        discretizer: The discretizer object used for preprocessing
        normalizer: The normalizer object used for preprocessing
    """
    # Set paths
    train_reader_path = os.path.join(args.data_path, args.task, 'train')
    val_reader_path = os.path.join(args.data_path, args.task, 'val')
    test_reader_path = os.path.join(args.data_path, args.task, 'test')

    # Initialize discretizer
    discretizer = Discretizer(
        timestep=args.timestep,
        store_masks=True,
        impute_strategy=args.imputation,
        config_path=os.path.join(args.data_path, 'discretizer_config.json')
    )

    # Initialize the appropriate reader based on the task
    if args.task == 'mortality':
        train_reader = InHospitalMortalityReader(train_reader_path,listfile=os.path.join(args.data_path, 'in-hospital-mortality/train_listfile.csv'))
        val_reader = InHospitalMortalityReader(val_reader_path,listfile=os.path.join(args.data_path, 'in-hospital-mortality/val_listfile.csv') )
        test_reader = InHospitalMortalityReader(test_reader_path,listfile=os.path.join(args.data_path, 'in-hospital-mortality/test_listfile.csv') )
    elif args.task == 'decompensation':
        train_reader = DecompensationReader(train_reader_path,listfile=os.path.join(args.data_path, 'decompensation/train_listfile.csv'))
        val_reader = DecompensationReader(val_reader_path,listfile=os.path.join(args.data_path, 'decompensation/val_listfile.csv'))
        test_reader = DecompensationReader(test_reader_path,listfile=os.path.join(args.data_path, 'decompensation/test_listfile.csv'))
    elif args.task == 'los':
        train_reader = LengthOfStayReader(train_reader_path,listfile=os.path.join(args.data_path, 'length-of-stay/train_listfile.csv'))
        val_reader = LengthOfStayReader(val_reader_path,listfile=os.path.join(args.data_path, 'length-of-stay/val_listfile.csv'))
        test_reader = LengthOfStayReader(test_reader_path,listfile=os.path.join(args.data_path, 'length-of-stay/test_listfile.csv'))
    else:
        raise ValueError(f"Unknown task: {args.task}")

    try:
        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    except IndexError:
        raise ValueError(
            "Could not read example 0 from train_reader to get header. Is the dataset empty or listfile incorrect?")


    print(f"    Discretizer header length: {len(discretizer_header)}")
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)
    base_normalizer_filename = "decomp_ts{}.input_str-previous.n1e5.start_time-zero.normalizer".format(args.timestep)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    normalizer_state_file_path = os.path.join(script_dir, base_normalizer_filename)
    normalizer.load_params(normalizer_state_file_path)
    normalizer_state_exists = True



    # Create batch generators
    train_data_gen = BatchGen(
        reader=train_reader,
        discretizer=discretizer,
        normalizer=normalizer,
        batch_size=args.batch_size,
        steps=None,
        shuffle=True,
        return_names=False
    )

    val_data_gen = BatchGen(
        reader=val_reader,
        discretizer=discretizer,
        normalizer=normalizer,
        batch_size=args.batch_size,
        steps=None,
        shuffle=False,
        return_names=False
    )

    test_data_gen = BatchGen(
        reader=test_reader,
        discretizer=discretizer,
        normalizer=normalizer,
        batch_size=args.batch_size,
        steps=None,
        shuffle=False,
        return_names=True  # Return names for test set
    )

    # If we need to build a normalizer from the training data
    if not normalizer_state_exists and args.mode == 'train':
        print('Building normalizer')

        # Read a chunk of training data
        chunk_data = common_utils.read_chunk(train_reader, 1000)

        # Extract a subset of variables (e.g., the continuous ones)
        # Assuming we have already defined continuous_channels above
        continuous_channels = [
            'Diastolic blood pressure',
            'Fraction inspired oxygen',
            'Glucose',
            'Heart Rate',
            'Mean blood pressure',
            'Oxygen saturation',
            'Respiratory rate',
            'Systolic blood pressure',
            'Temperature'
        ]

        # Extract the indices of continuous channels
        cont_channel_indices = []
        for i, channel in enumerate(discretizer._id_to_channel):
            if channel in continuous_channels:
                cont_channel_indices.append(i)

        # Extract and preprocess the normalizer data
        normalizer_data = []
        for i in range(len(chunk_data['X'])):
            processed_data, _ = discretizer.transform(chunk_data['X'][i], end=chunk_data['t'][i])
            normalizer_data.append(processed_data[:, cont_channel_indices])

        # Update normalizer with the data
        normalizer._feed_data(np.concatenate(normalizer_data, axis=0))
        normalizer._save_params(normalizer_state)

    return train_data_gen, val_data_gen, test_data_gen, discretizer, normalizer


def get_dataset_from_generator(generator, output_types, output_shapes):
    """
    Convert a BatchGen generator to a tf.data.Dataset

    Args:
        generator: A generator yielding batches
        output_types: Types of the output tensor
        output_shapes: Shapes of the output tensor

    Returns:
        A tf.data.Dataset
    """

    def gen_wrapper():
        for batch in generator:
            yield batch

    return tf.data.Dataset.from_generator(
        gen_wrapper,
        output_types=output_types,
        output_shapes=output_shapes
    )


def train(args):
    """
    Train a model for the specified task.

    Args:
        args: Command-line arguments
    """
    print(f"Starting training for task: {args.task} using {args.network} network")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_gen, val_gen, test_gen, discretizer, normalizer = load_data(args)

    # Determine input shape from a sample batch
    sample_batch = next(train_gen)
    input_shape = sample_batch[0].shape[1:]  # (timesteps, features)

    # Create model
    model, loss, metrics = create_network(args, input_shape)
    model.summary()

    # Configure optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr,
            beta_1=args.beta_1
        )
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=args.lr
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=args.lr,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Define callbacks
    callbacks = [
        # Checkpoint to save the best model
        ModelCheckpoint(
            filepath=os.path.join(args.output_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Convert generator to tf.data.Dataset for better performance
    output_types = (tf.float32, tf.float32)  # (x, y) types

    # Define output shape based on task
    if args.task in ['mortality', 'decompensation']:
        y_shape = tf.TensorShape([None, 1])  # binary classification
    elif args.task == 'los':
        y_shape = tf.TensorShape([None, 1])  # regression
    elif args.task == 'phenotyping':
        y_shape = tf.TensorShape([None, 25])  # multi-label

    output_shapes = (
        tf.TensorShape([None, input_shape[0], input_shape[1]]),  # x shape: (batch, timesteps, features)
        y_shape
    )

    train_dataset = get_dataset_from_generator(train_gen, output_types, output_shapes)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = get_dataset_from_generator(val_gen, output_types, output_shapes)

    # Train the model
    print("Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=args.verbose
    )

    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    # Save the final model
    model.save(os.path.join(args.output_dir, 'final_model.h5'))

    # Evaluate on test set
    if args.evaluate:
        evaluate(args, model, test_gen)


def evaluate(args, model=None, test_gen=None):
    """
    Evaluate a model for the specified task.

    Args:
        args: Command-line arguments
        model: Optional. The model to evaluate. If None, load from file.
        test_gen: Optional. The test data generator. If None, create one.
    """
    # If model is not provided, load it
    if model is None:
        model_path = os.path.join(args.output_dir, 'best_model.h5')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.output_dir, 'final_model.h5')

        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")

    # If test generator is not provided, create one
    if test_gen is None:
        _, _, test_gen, _, _ = load_data(args)

    # Dictionary to store results with names and timestamps
    results = {'names': [], 'predictions': [], 'true_values': [], 'timestamps': []}

    # Collect predictions and true values
    for batch in test_gen:
        if isinstance(batch, dict):  # If return_names=True is used
            x = batch['data'][0]
            y_true = batch['data'][1]
            names = batch['names']
            ts = batch['ts']
        else:
            x, y_true = batch
            names = None
            ts = None

        # Generate predictions
        y_pred = model.predict(x)

        # Store predictions, true values, and names
        if names is not None:
            results['names'].extend(names)
            results['timestamps'].extend(ts)

        results['predictions'].extend(y_pred)
        results['true_values'].extend(y_true)

    # Convert lists to numpy arrays
    results['predictions'] = np.array(results['predictions'])
    results['true_values'] = np.array(results['true_values'])

    # Save detailed predictions if names are available
    if results['names']:
        prediction_path = os.path.join(args.output_dir, f'predictions_{args.task}.csv')
        if hasattr(common_utils, 'save_results'):
            common_utils.save_results(
                results['names'],
                results['timestamps'],
                results['predictions'],
                results['true_values'],
                prediction_path
            )
            print(f"Saved detailed predictions to {prediction_path}")
        else:
            # Fallback if save_results function doesn't exist
            with open(prediction_path, 'w') as f:
                f.write('patient,timestamp,prediction,ground_truth\n')
                for name, ts, pred, true in zip(
                        results['names'],
                        results['timestamps'],
                        results['predictions'],
                        results['true_values']
                ):
                    f.write(f"{name},{ts},{pred[0]},{true[0]}\n")
            print(f"Saved detailed predictions to {prediction_path}")

    # Compute and print task-specific metrics
    if args.task == 'mortality' or args.task == 'decompensation':
        metrics = print_metrics_binary(results['true_values'], results['predictions'])
    elif args.task == 'los':
        metrics = print_metrics_regression(results['true_values'], results['predictions'])
    elif args.task == 'phenotyping':
        metrics = print_metrics_multilabel(results['true_values'], results['predictions'])

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'metrics_{args.task}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Train a model for clinical predictions')

    # Add common arguments from the original repo
    common_utils.add_common_arguments(parser)

    # Add task-specific arguments
    parser.add_argument('--task', type=str, required=True,
                        choices=['mortality', 'decompensation', 'los', 'phenotyping'],
                        help='The prediction task')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the MIMIC-III data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save outputs')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on test set after training')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Which GPU to use')

    # Additional arguments for transformer network
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads in transformer')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of transformer blocks')

    # Additional arguments for CNN network
    parser.add_argument('--filters', type=int, default=64,
                        help='Number of filters in CNN')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3, 5, 7],
                        help='Kernel sizes for CNN layers')

    # Mode argument to control flow
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help='Mode to run the script in')

    # Parse arguments
    args = parser.parse_args()

    # Set GPU if specified
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Run mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)


if __name__ == "__main__":
    main()