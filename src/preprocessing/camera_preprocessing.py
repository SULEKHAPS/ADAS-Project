import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class CameraPreprocessor:
    """
    Class for preprocessing camera data and detecting objects using transfer learning
    with a pre-trained ResNet50 model fine-tuned on custom data.
    """
    def __init__(self, model_path=None, num_classes=10, confidence_threshold=0.5):
        """
        Initialize the camera preprocessor.
        
        Args:
            model_path (str): Path to the saved model weights. If None, a new model will be created.
            num_classes (int): Number of object classes to detect.
            confidence_threshold (float): Threshold for object detection confidence.
        """
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        self.class_names = [
            'car', 'truck', 'pedestrian', 'cyclist', 'motorcycle', 
            'bus', 'traffic_sign', 'traffic_light', 'animal', 'obstacle'
        ]
        
        # Create or load the model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            self.model = self._create_model()
    
    def _create_model(self):
        """
        Create a transfer learning model based on ResNet50 for object detection.
        
        Returns:
            Model: The compiled transfer learning model.
        """
        # Load the pre-trained ResNet50 model without the top classification layer
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add custom layers for our specific task
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='sigmoid')(x)  # Multi-label classification
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _load_model(self, model_path):
        """
        Load a pre-trained model from disk.
        
        Args:
            model_path (str): Path to the saved model weights.
            
        Returns:
            Model: The loaded model.
        """
        model = self._create_model()
        model.load_weights(model_path)
        return model
    
    def preprocess_image(self, img):
        """
        Preprocess an image for the model.
        
        Args:
            img (numpy.ndarray): Input image in BGR format (OpenCV default).
            
        Returns:
            numpy.ndarray: Preprocessed image ready for the model.
        """
        # Convert BGR to RGB (if the image is from OpenCV)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize the image to the required input size
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Convert to a tensor and preprocess for ResNet50
        img_array = image.img_to_array(img_resized)
        img_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_expanded)
        
        return img_preprocessed
    
    def detect_objects(self, img):
        """
        Detect objects in an image.
        
        Args:
            img (numpy.ndarray): Input image in BGR format (OpenCV default).
            
        Returns:
            list: List of detected objects with class names and confidence scores.
        """
        # Preprocess the image
        preprocessed_img = self.preprocess_image(img)
        
        # Make predictions
        predictions = self.model.predict(preprocessed_img)[0]
        
        # Filter predictions based on confidence threshold
        detected_objects = []
        for i, confidence in enumerate(predictions):
            if confidence >= self.confidence_threshold:
                detected_objects.append({
                    'class_name': self.class_names[i],
                    'confidence': float(confidence),
                    'bounding_box': None  # In a real implementation, we would need a separate model for bounding boxes
                })
        
        return detected_objects
    
    def process_video_frame(self, frame):
        """
        Process a single video frame to detect objects.
        
        Args:
            frame (numpy.ndarray): Input video frame in BGR format.
            
        Returns:
            tuple: (processed_frame, detected_objects)
        """
        # Create a copy of the frame for visualization
        visualization_frame = frame.copy()
        
        # Detect objects in the frame
        detected_objects = self.detect_objects(frame)
        
        # Visualize the detected objects (in a real implementation, we would draw bounding boxes)
        for obj in detected_objects:
            # Add text labels for detected objects
            cv2.putText(
                visualization_frame,
                f"{obj['class_name']}: {obj['confidence']:.2f}",
                (10, 30),  # Position would be based on bounding box in real implementation
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        return visualization_frame, detected_objects
    
    def train(self, train_data, validation_data, epochs=10, batch_size=32):
        """
        Train the model on custom data.
        
        Args:
            train_data: Training data generator or dataset.
            validation_data: Validation data generator or dataset.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            
        Returns:
            History: Training history.
        """
        # Unfreeze some layers for fine-tuning
        for layer in self.model.layers[-20:]:  # Unfreeze the last 20 layers
            layer.trainable = True
        
        # Recompile the model with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def save_model(self, model_path):
        """
        Save the model weights to disk.
        
        Args:
            model_path (str): Path to save the model weights.
        """
        self.model.save_weights(model_path)
        print(f"Model saved to {model_path}")


# Example usage
if __name__ == "__main__":
    # Initialize the camera preprocessor
    camera_processor = CameraPreprocessor()
    
    # Example: Process a single image
    image_path = "path/to/test/image.jpg"
    if os.path.exists(image_path):
        # Load and process the image
        img = cv2.imread(image_path)
        processed_img, objects = camera_processor.process_video_frame(img)
        
        # Display results
        print(f"Detected objects: {objects}")
        cv2.imshow("Processed Image", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Image not found: {image_path}")