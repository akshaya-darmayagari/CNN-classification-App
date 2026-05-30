# evaluation.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_test_data():
    """Loads and returns test data partition."""
    _, (x_test, y_test) = cifar10.load_data()
    x_test_normalized = x_test.astype("float32") / 255.0
    return x_test_normalized, y_test

def main():
    model_path = 'cifar10_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Run model_training.py first.")
        return

    print("Loading test dataset and model artifact...")
    x_test, y_test_raw = get_test_data()
    y_test_categorical = to_categorical(y_test_raw, 10)
    
    model = tf.keras.models.load_model(model_path)
    
    # 1. Base evaluation metrics
    print("\nEvaluating model on the test dataset...")
    test_loss, test_acc = model.evaluate(x_test, y_test_categorical, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 2. Detailed Classification Report
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\nDetailed Classification Metrics:")
    print(classification_report(y_test_raw, predicted_classes, target_names=CLASS_NAMES))
    
    # 3. Save Confusion Matrix Plot
    print("Generating confusion matrix visual plot...")
    cm = confusion_matrix(y_test_raw, predicted_classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    
    plot_filename = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Confusion matrix image saved as '{plot_filename}'.")

if __name__ == '__main__':
    main()