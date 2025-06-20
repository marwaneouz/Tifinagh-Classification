# src/train.py

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from preprocess import preprocess_data
from model import MulticlassNeuralNetwork

def main():
    data_dir = os.path.join(os.getcwd(), "data", "tifinagh-images")
    print("Loading dataset...")
    df = preprocess.load_dataset(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, label_encoder = preprocess_data(df)

    print("Building model...")
    input_size = X_train.shape[1]
    layer_sizes = [input_size, 64, 32, num_classes]
    model = MulticlassNeuralNetwork(layer_sizes, learning_rate=0.01)

    print("Training model...")
    train_losses, val_losses, train_accuracies, val_accuracies = model.train(
        X_train, y_train, X_val, y_val, epochs=100, batch_size=32
    )

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("\nClassification Report (Test Set):")
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=label_encoder.classes_))

    print("Plotting results...")
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(train_accuracies, label="Train Accuracy")
    ax2.plot(val_accuracies, label="Validation Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from preprocess import preprocess_data
    main()