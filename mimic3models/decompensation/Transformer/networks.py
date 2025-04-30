import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np


def create_lstm_network(input_shape, task, depth=1, dropout=0.0, rec_dropout=0.0, 
                        dim=256, batch_norm=False, l1=0, l2=0):
    """
    Create a bidirectional LSTM network for time series predictions.
    
    Parameters:
    - input_shape: Shape of input data (timesteps, features)
    - task: The prediction task (mortality, decompensation, los, phenotyping)
    - depth: Number of stacked LSTM layers
    - dropout: Dropout rate for dense layers
    - rec_dropout: Recurrent dropout rate for LSTM layers
    - dim: Dimension of LSTM hidden units
    - batch_norm: Whether to use batch normalization
    - l1: L1 regularization coefficient
    - l2: L2 regularization coefficient
    
    Returns:
    - model: A compiled Keras model
    """
    # Define the number of output classes based on task
    if task == 'mortality' or task == 'decompensation':
        output_dim = 1  # Binary classification (single sigmoid)
    elif task == 'los':
        output_dim = 1  # Regression
    elif task == 'phenotyping':
        output_dim = 25  # Multi-label classification
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create regularizer
    reg = regularizers.l1_l2(l1=l1, l2=l2)
    
    # Input layers
    inputs = layers.Input(shape=input_shape)
    
    # Masking layer to handle variable sequence lengths
    masked = layers.Masking(mask_value=0.0)(inputs)
    
    # LSTM layers
    x = masked
    for i in range(depth):
        x = layers.Bidirectional(
            layers.LSTM(
                dim, 
                return_sequences=(i < depth-1),  # Return sequences for all except last layer
                dropout=dropout,
                recurrent_dropout=rec_dropout,
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
                bias_regularizer=reg
            )
        )(x)
        
        if batch_norm:
            x = layers.BatchNormalization()(x)
    
    # Task-specific output layers
    if task == 'mortality' or task == 'decompensation':
        outputs = layers.Dense(output_dim, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task == 'los':
        outputs = layers.Dense(output_dim, activation='linear', name='output')(x)
        loss = 'mse'
        metrics = ['mae']
    elif task == 'phenotyping':
        outputs = layers.Dense(output_dim, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    
    # Create and compile model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model, loss, metrics


def create_transformer_network(input_shape, task, num_heads=8, num_blocks=4, dim=256, 
                              dropout=0.1, l1=0, l2=0, ff_dim=None):
    """
    Create a Transformer model for time series forecasting.
    
    Parameters:
    - input_shape: Shape of input data (timesteps, features)
    - task: The prediction task (mortality, decompensation, los, phenotyping)
    - num_heads: Number of attention heads
    - num_blocks: Number of transformer blocks
    - dim: Dimension of the model
    - dropout: Dropout rate
    - l1: L1 regularization coefficient
    - l2: L2 regularization coefficient
    - ff_dim: Feed forward dimension, if None uses 4*dim
    
    Returns:
    - model: A Keras model
    - loss: Loss function name
    - metrics: List of metrics
    """
    # Define the number of output classes based on task
    if task == 'mortality' or task == 'decompensation':
        output_dim = 1  # Binary classification (single sigmoid)
    elif task == 'los':
        output_dim = 1  # Regression
    elif task == 'phenotyping':
        output_dim = 25  # Multi-label classification
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Set feed forward dimension if not provided
    if ff_dim is None:
        ff_dim = dim * 4
    
    # Create regularizer
    reg = regularizers.l1_l2(l1=l1, l2=l2)
    
    # Input layers
    inputs = layers.Input(shape=input_shape)
    
    # Masking layer to handle variable sequence lengths
    masked = layers.Masking(mask_value=0.0)(inputs)
    
    # Position encoding
    positions = positional_encoding(input_shape[0], input_shape[1])
    x = layers.Add()([masked, positions])
    
    # Transformer encoder blocks
    for _ in range(num_blocks):
        x = transformer_encoder_layer(x, dim=dim, num_heads=num_heads, dropout=dropout, l1=l1, l2=l2, ff_dim=ff_dim)
    
    # Global pooling for sequence reduction
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final dense layer with dropout
    x = layers.Dense(dim, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    
    # Task-specific output layers
    if task == 'mortality' or task == 'decompensation':
        outputs = layers.Dense(output_dim, activation='sigmoid', name='output', kernel_regularizer=reg)(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task == 'los':
        outputs = layers.Dense(output_dim, activation='linear', name='output', kernel_regularizer=reg)(x)
        loss = 'mse'
        metrics = ['mae']
    elif task == 'phenotyping':
        outputs = layers.Dense(output_dim, activation='sigmoid', name='output', kernel_regularizer=reg)(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model, loss, metrics


def positional_encoding(max_seq_len, d_model):
    """
    Compute positional encoding for transformer model
    
    Args:
        max_seq_len: Maximum sequence length
        d_model: Dimension of the model
        
    Returns:
        pos_encoding: Positional encoding tensor of shape (1, max_seq_len, d_model)
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((max_seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(positions * div_term)
    pos_encoding[:, 1::2] = np.cos(positions * div_term)
    
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)


def transformer_encoder_layer(inputs, dim, num_heads, dropout=0.1, l1=0, l2=0, ff_dim=None):
    """
    Create a Transformer encoder layer
    
    Args:
        inputs: Input tensor
        dim: Dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout rate
        l1: L1 regularization coefficient
        l2: L2 regularization coefficient
        ff_dim: Feed forward dimension, if None uses 4*dim
        
    Returns:
        Output tensor of the encoder layer
    """
    if ff_dim is None:
        ff_dim = dim * 4
    
    # Create regularizer
    reg = regularizers.l1_l2(l1=l1, l2=l2)
    
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=dim // num_heads, 
        kernel_regularizer=reg
    )(inputs, inputs)
    
    # Add & Norm
    attention_output = layers.Dropout(dropout)(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed Forward Network
    ffn_output = layers.Dense(ff_dim, activation='relu', kernel_regularizer=reg)(x)
    ffn_output = layers.Dense(dim, kernel_regularizer=reg)(ffn_output)
    
    # Add & Norm
    ffn_output = layers.Dropout(dropout)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)


def create_cnn_network(input_shape, task, filters=64, kernel_sizes=[3, 5, 7], 
                      dropout=0.0, dim=256, l1=0, l2=0):
    """
    Create a CNN network for time series predictions.
    
    Parameters:
    - input_shape: Shape of input data (timesteps, features)
    - task: The prediction task (mortality, decompensation, los, phenotyping)
    - filters: Number of convolutional filters
    - kernel_sizes: List of kernel sizes for different convolutional layers
    - dropout: Dropout rate
    - dim: Dimension of dense layer
    - l1: L1 regularization coefficient
    - l2: L2 regularization coefficient
    
    Returns:
    - model: A Keras model
    - loss: Loss function name
    - metrics: List of metrics
    """
    # Define the number of output classes based on task
    if task == 'mortality' or task == 'decompensation':
        output_dim = 1  # Binary classification (single sigmoid)
    elif task == 'los':
        output_dim = 1  # Regression
    elif task == 'phenotyping':
        output_dim = 25  # Multi-label classification
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create regularizer
    reg = regularizers.l1_l2(l1=l1, l2=l2)
    
    # Input layers
    inputs = layers.Input(shape=input_shape)
    
    # Masking layer to handle variable sequence lengths
    masked = layers.Masking(mask_value=0.0)(inputs)
    
    # CNN layers with different kernel sizes
    conv_outputs = []
    for kernel_size in kernel_sizes:
        conv = layers.Conv1D(
            filters=filters, 
            kernel_size=kernel_size,
            padding='same', 
            activation='relu',
            kernel_regularizer=reg
        )(masked)
        pool = layers.GlobalMaxPooling1D()(conv)
        conv_outputs.append(pool)
    
    # Concatenate all CNN outputs
    if len(conv_outputs) > 1:
        x = layers.Concatenate()(conv_outputs)
    else:
        x = conv_outputs[0]
    
    # Dense layer
    x = layers.Dense(dim, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    
    # Task-specific output layers
    if task == 'mortality' or task == 'decompensation':
        outputs = layers.Dense(output_dim, activation='sigmoid', name='output', kernel_regularizer=reg)(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task == 'los':
        outputs = layers.Dense(output_dim, activation='linear', name='output', kernel_regularizer=reg)(x)
        loss = 'mse'
        metrics = ['mae']
    elif task == 'phenotyping':
        outputs = layers.Dense(output_dim, activation='sigmoid', name='output', kernel_regularizer=reg)(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model, loss, metrics


def create_network(args, input_shape):
    """
    Create a model based on network type
    
    Parameters:
    - args: Command-line arguments
    - input_shape: Shape of input data
    
    Returns:
    - model: A Keras model
    - loss: Loss function name
    - metrics: List of metrics
    """
    network_type = args.network.lower()
    
    if network_type == 'lstm':
        return create_lstm_network(
            input_shape=input_shape,
            task=args.task,
            depth=args.depth,
            dropout=args.dropout,
            rec_dropout=args.rec_dropout,
            dim=args.dim,
            batch_norm=args.batch_norm,
            l1=args.l1,
            l2=args.l2
        )
    elif network_type == 'transformer':
        return create_transformer_network(
            input_shape=input_shape,
            task=args.task,
            num_heads=args.num_heads if hasattr(args, 'num_heads') else 8,
            num_blocks=args.num_blocks if hasattr(args, 'num_blocks') else 4,
            dim=args.dim,
            dropout=args.dropout,
            l1=args.l1,
            l2=args.l2
        )
    elif network_type == 'cnn':
        kernel_sizes = [3, 5, 7]  # Default kernel sizes
        if hasattr(args, 'kernel_sizes'):
            kernel_sizes = args.kernel_sizes
        
        return create_cnn_network(
            input_shape=input_shape,
            task=args.task,
            filters=args.filters if hasattr(args, 'filters') else 64,
            kernel_sizes=kernel_sizes,
            dropout=args.dropout,
            dim=args.dim,
            l1=args.l1,
            l2=args.l2
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")
