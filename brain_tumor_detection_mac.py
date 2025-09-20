# brain_tumor_detection_mac.py
import os
import numpy as np
import random
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMAGE_SIZE = 128
BATCH_SIZE = 20
EPOCHS = 5
LEARNING_RATE = 0.0001

# Hardcoded path as requested
BASE_DATA_PATH = '/Users/abnv/Desktop/testing/Brain-Tumor-Detection-Using-Deep-Learning-MRI-Images-Detection-Using-Computer-Vision/MRI IMAGES'

class BrainTumorDetector:
    def __init__(self):
        self.model = None
        self.class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
        self.history = None
        
    def setup_directories(self, base_path):
        """Setup directory paths for training and testing data"""
        train_dir = os.path.join(base_path, 'Training')
        test_dir = os.path.join(base_path, 'Testing')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found at: {train_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Testing directory not found at: {test_dir}")
            
        return train_dir, test_dir
    
    def load_data_paths(self, data_dir):
        """Load image paths and labels from directory"""
        paths = []
        labels = []
        
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        paths.append(os.path.join(label_dir, image_name))
                        labels.append(label)
        
        return paths, labels
    
    def display_sample_images(self, paths, labels, num_samples=10):
        """Display a sample of images from the dataset"""
        if len(paths) < num_samples:
            num_samples = len(paths)
            
        random_indices = random.sample(range(len(paths)), num_samples)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(random_indices):
            img = Image.open(paths[idx])
            img = img.resize((224, 224))
            
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Label: {labels[idx]}", fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def augment_image(self, image):
        """Apply random augmentations to an image"""
        image = Image.fromarray(np.uint8(image))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        return image
    
    def open_images(self, paths):
        """Load and augment images from paths"""
        images = []
        for path in paths:
            image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            image = self.augment_image(image)
            images.append(image)
        return np.array(images)
    
    def encode_label(self, labels):
        """Convert string labels to integer encoding"""
        encoded = [self.class_labels.index(label) for label in labels]
        return np.array(encoded)
    
    def datagen(self, paths, labels, batch_size=12, epochs=1):
        """Data generator for batching"""
        for _ in range(epochs):
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                batch_images = self.open_images(batch_paths)
                batch_labels_encoded = self.encode_label(batch_labels)
                
                yield batch_images, batch_labels_encoded
    
    def build_model(self):
        """Build the VGG16-based model"""
        base_model = VGG16(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), 
            include_top=False, 
            weights='imagenet'
        )
        
        # Freeze all layers initially
        for layer in base_model.layers:
            layer.trainable = False
        
        # Unfreeze last few layers for fine-tuning
        base_model.layers[-2].trainable = True
        base_model.layers[-3].trainable = True
        base_model.layers[-4].trainable = True
        
        # Build the complete model
        model = Sequential()
        model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(base_model)
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.class_labels), activation='softmax'))
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        
        self.model = model
        print("Model built and compiled successfully.")
        return model
    
    def train(self, train_paths, train_labels):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        steps = int(len(train_paths) / BATCH_SIZE)
        
        self.history = self.model.fit(
            self.datagen(train_paths, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS),
            epochs=EPOCHS,
            steps_per_epoch=steps
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history (accuracy and loss)"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        plt.figure(figsize=(8, 4))
        plt.grid(True)
        plt.plot(self.history.history['sparse_categorical_accuracy'], '.g-', linewidth=2, label='Accuracy')
        plt.plot(self.history.history['loss'], '.r-', linewidth=2, label='Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.xticks([x for x in range(EPOCHS)])
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, test_paths, test_labels):
        """Evaluate the model on test data"""
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return
        
        test_images = self.open_images(test_paths)
        test_labels_encoded = self.encode_label(test_labels)
        
        # Predict
        test_predictions = self.model.predict(test_images)
        predicted_classes = np.argmax(test_predictions, axis=1)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(test_labels_encoded, predicted_classes, target_names=self.class_labels))
        
        # Confusion matrix
        conf_matrix = confusion_matrix(test_labels_encoded, predicted_classes)
        print("Confusion Matrix:")
        print(conf_matrix)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()
        plt.show()
        
        # ROC Curve
        test_labels_bin = label_binarize(test_labels_encoded, classes=np.arange(len(self.class_labels)))
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(len(self.class_labels)):
            fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 8))
        for i in range(len(self.class_labels)):
            plt.plot(fpr[i], tpr[i], label=f'{self.class_labels[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='brain_tumor_model.h5'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save. Please train the model first.")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='brain_tumor_model.h5'):
        """Load a trained model"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found.")
            return
        
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def detect_and_display(self, img_path):
        """Detect tumor in an image and display results"""
        if self.model is None:
            print("Model not loaded. Please load or train a model first.")
            return
        
        try:
            # Load and preprocess the image
            img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = np.max(predictions, axis=1)[0]
            
            # Determine result
            if self.class_labels[predicted_class_index] == 'notumor':
                result = "No Tumor"
            else:
                result = f"Tumor: {self.class_labels[predicted_class_index]}"
            
            # Display the image with prediction
            plt.figure(figsize=(6, 6))
            plt.imshow(load_img(img_path))
            plt.axis('off')
            plt.title(f"{result} (Confidence: {confidence_score * 100:.2f}%)")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error processing the image: {str(e)}")

def main():
    # Initialize the detector
    detector = BrainTumorDetector()
    
    print(f"Using hardcoded data path: {BASE_DATA_PATH}")
    
    try:
        # Setup directories
        train_dir, test_dir = detector.setup_directories(BASE_DATA_PATH)
        
        # Load data paths
        train_paths, train_labels = detector.load_data_paths(train_dir)
        test_paths, test_labels = detector.load_data_paths(test_dir)
        
        # Shuffle the data
        train_paths, train_labels = shuffle(train_paths, train_labels)
        test_paths, test_labels = shuffle(test_paths, test_labels)
        
        print(f"Loaded {len(train_paths)} training images and {len(test_paths)} test images.")
        
        # Display sample images
        print("Displaying sample training images...")
        detector.display_sample_images(train_paths, train_labels)
        
        # Build and train the model
        print("Building model...")
        detector.build_model()
        
        print("Training model...")
        detector.train(train_paths, train_labels)
        
        # Plot training history
        detector.plot_training_history()
        
        # Evaluate the model
        print("Evaluating model...")
        detector.evaluate(test_paths, test_labels)
        
        # Save the model
        detector.save_model()
        
        # Test with sample images
        print("Testing with sample images...")
        
        # Test with sample images from each category
        sample_images = []
        for category in detector.class_labels:
            category_test_dir = os.path.join(test_dir, category)
            if os.path.exists(category_test_dir) and os.listdir(category_test_dir):
                sample_image = os.path.join(category_test_dir, os.listdir(category_test_dir)[0])
                sample_images.append(sample_image)
        
        for img_path in sample_images:
            if os.path.exists(img_path):
                print(f"Processing {img_path}...")
                detector.detect_and_display(img_path)
            else:
                print(f"Image not found: {img_path}")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure:")
        print("1. The hardcoded path to your MRI IMAGES directory is correct")
        print("2. The directory contains 'Training' and 'Testing' subdirectories")
        print("3. Each subdirectory contains folders for each class (pituitary, glioma, notumor, meningioma)")

if __name__ == "__main__":
    main()