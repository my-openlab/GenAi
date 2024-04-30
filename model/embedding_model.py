import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Layer

def create_embedding_model(input_shape):
    model = tf.keras.Sequential(name=f'feature_embedding')
    
    # Convolutional Blocks
    num_channels = 24
    for _ in range(5):
        model.add(Conv2D(num_channels, (1, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(Conv2D(num_channels, (3, 1), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(num_channels, (1, 3), activation='relu', padding='same'))
        model.add(Conv2D(num_channels, (3, 1), activation='relu', padding='same'))
        num_channels += 24
        if num_channels > 96:
            num_channels = 96
        
    
    # Final Convolutional Block
    model.add(Conv2D(num_channels, (3, 1), activation='relu', padding='same'))
    
    # Classification Block
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1, (1, 1)))  # Change this line to Dense(1) if fully connected layer is preferred
    
    return model


# Define input shape
sr = 16000 # sampling rate, in hz
record_duration = 1.92 # in seconds
sample_length, frequency_dimension = int(record_duration*sr), 32
embedding_length = 1 + (sample_length - 12400) // 1280

input_shape = (embedding_length, frequency_dimension, 1)  # Update with actual values
model = create_embedding_model(input_shape)

# Print model summary
model.summary()
