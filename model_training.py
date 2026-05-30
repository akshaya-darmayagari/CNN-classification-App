# model_training.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def get_data():
    """Fetches and normalizes training/validation data from TensorFlow."""
    print("Fetching CIFAR-10 training data...")
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    
    # Scale values to [0, 1] range
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    
    # Convert integer targets to categorical vectors
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    
    return x_train, y_train, x_val, y_val

def compile_cnn():
    """Initializes and compiles the CNN architecture."""
    model = models.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second conv block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fully connected block
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    x_train, y_train, x_val, y_val = get_data()
    model = compile_cnn()
    model.summary()
    
    print("\nTraining model...")
    # Train for 15 epochs
    model.fit(
        x_train, y_train, 
        epochs=15, 
        batch_size=64, 
        validation_data=(x_val, y_val)
    )
    
    model_name = "cifar10_model.h5"
    model.save(model_name)
    print(f"\nModel training finished. Saved artifact as '{model_name}'.")

if __name__ == '__main__':
    main()