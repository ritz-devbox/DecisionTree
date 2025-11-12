"""
Placement Prediction System using Decision Tree Classifier
Improved version with better code structure, error handling, and best practices
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from pathlib import Path
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlacementPredictor:
    """Main class for placement prediction using Decision Tree Classifier"""
    
    def __init__(self, data_file: str = "a.csv", model_file: str = "placement_model.pkl"):
        self.data_file = data_file
        self.model_file = model_file
        self.model: Optional[DecisionTreeClassifier] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = ['Aggregate', 'Backlogs', '10th_agg', '12th_agg', 'Workshops', 'Languages']
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess data from CSV file"""
        try:
            # Use pandas for better CSV handling
            df = pd.read_csv(self.data_file)
            
            # Assuming last column is target, rest are features
            X = df.iloc[:, 1:7].values  # Features (columns 1-6)
            y = df.iloc[:, -1].values   # Target (last column)
            
            # Remove any newline characters from target
            y = np.array([str(label).strip().replace('\n', '') for label in y])
            
            logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
            return X, y
            
        except FileNotFoundError:
            logger.error(f"Data file {self.data_file} not found")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42) -> dict:
        """Train the decision tree classifier and return metrics"""
        try:
            X, y = self.load_data()
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Create and train model with better parameters
            self.model = DecisionTreeClassifier(
                max_depth=10,           # Prevent overfitting
                min_samples_split=5,    # Minimum samples to split
                min_samples_leaf=2,     # Minimum samples in leaf
                random_state=42,        # For reproducibility
                criterion='gini'        # or 'entropy'
            )
            
            self.model.fit(self.X_train, self.y_train)
            
            # Evaluate model
            train_score = self.model.score(self.X_train, self.y_train)
            test_score = self.model.score(self.X_test, self.y_test)
            
            # Get predictions for detailed metrics
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'accuracy': accuracy,
                'classification_report': classification_report(self.y_test, y_pred)
            }
            
            logger.info(f"Model trained - Train Accuracy: {train_score:.2%}, Test Accuracy: {test_score:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def save_model(self) -> bool:
        """Save the trained model to disk"""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train the model first.")
            joblib.dump(self.model, self.model_file)
            logger.info(f"Model saved to {self.model_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load a pre-trained model from disk"""
        try:
            if os.path.exists(self.model_file):
                self.model = joblib.load(self.model_file)
                logger.info(f"Model loaded from {self.model_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, features: List[float]) -> Tuple[str, float]:
        """
        Predict placement for given features
        
        Args:
            features: List of [Aggregate, Backlogs, 10th_agg, 12th_agg, Workshops, Languages]
            
        Returns:
            Tuple of (prediction, confidence/probability)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        try:
            # Validate input
            if len(features) != 6:
                raise ValueError(f"Expected 6 features, got {len(features)}")
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features_array)[0]
            
            # Get prediction probability
            probabilities = self.model.predict_proba(features_array)[0]
            confidence = max(probabilities)
            
            return str(prediction), confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise


class PlacementGUI:
    """GUI application for placement prediction"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Training and Placement Cell - Placement Prediction System')
        self.root.geometry('800x700')
        self.root.configure(bg='lightblue')
        
        self.predictor = PlacementPredictor()
        
        # Try to load existing model, otherwise train new one
        if not self.predictor.load_model():
            try:
                metrics = self.predictor.train_model()
                self.predictor.save_model()
                logger.info("New model trained and saved")
            except Exception as e:
                messagebox.showerror("Error", f"Could not train model: {e}")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(
            self.root,
            text='STUDENT PLACEMENT PREDICTION SYSTEM',
            bg="lightblue",
            fg="black",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=20)
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg='lightblue')
        main_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Input fields
        self.entries = {}
        fields = [
            ('Roll Number', 'roll_number'),
            ('Aggregate (%)', 'aggregate'),
            ('Backlogs', 'backlogs'),
            ('10th Aggregate (%)', 'tenth_agg'),
            ('12th Aggregate (%)', 'twelfth_agg'),
            ('Workshops', 'workshops'),
            ('Programming Languages', 'languages')
        ]
        
        for i, (label_text, key) in enumerate(fields):
            label = tk.Label(main_frame, text=label_text, bg="lightblue", font=("Arial", 10, "bold"))
            label.grid(row=i, column=0, sticky='w', pady=5, padx=10)
            
            entry = tk.Entry(main_frame, width=30, font=("Arial", 10))
            entry.grid(row=i, column=1, pady=5, padx=10, sticky='w')
            self.entries[key] = entry
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='lightblue')
        button_frame.grid(row=len(fields), column=0, columnspan=2, pady=20)
        
        submit_btn = tk.Button(
            button_frame,
            text='Predict Placement',
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.predict_placement,
            padx=20,
            pady=10
        )
        submit_btn.pack(side='left', padx=10)
        
        clear_btn = tk.Button(
            button_frame,
            text='Clear',
            bg="#f44336",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.clear_fields,
            padx=20,
            pady=10
        )
        clear_btn.pack(side='left', padx=10)
        
        # Result area
        result_label = tk.Label(main_frame, text='Prediction Result:', bg="lightblue", font=("Arial", 12, "bold"))
        result_label.grid(row=len(fields)+1, column=0, columnspan=2, pady=(20, 5), sticky='w', padx=10)
        
        self.result_text = tk.Text(
            main_frame,
            width=50,
            height=5,
            bg="lightyellow",
            wrap=tk.WORD,
            font=("Arial", 11)
        )
        self.result_text.grid(row=len(fields)+2, column=0, columnspan=2, pady=5, padx=10, sticky='w')
        
        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text='Ready',
            bg="lightblue",
            font=("Arial", 9),
            anchor='w'
        )
        self.status_label.grid(row=len(fields)+3, column=0, columnspan=2, pady=10, padx=10, sticky='w')
    
    def validate_inputs(self) -> Optional[List[float]]:
        """Validate and convert user inputs"""
        try:
            features = []
            
            # Roll number validation (optional but if provided, should be 7-8 digits)
            roll_number = self.entries['roll_number'].get().strip()
            if roll_number and (len(roll_number) < 7 or len(roll_number) > 8):
                messagebox.showerror("Validation Error", "Roll number should be 7-8 digits")
                return None
            
            # Aggregate validation
            aggregate = float(self.entries['aggregate'].get())
            if aggregate < 0 or aggregate > 100:
                messagebox.showerror("Validation Error", "Aggregate must be between 0 and 100")
                return None
            if aggregate < 50:
                messagebox.showwarning("Warning", "Aggregate is below 50%. Student may not be eligible.")
            features.append(aggregate)
            
            # Backlogs validation
            backlogs = float(self.entries['backlogs'].get())
            if backlogs < 0:
                messagebox.showerror("Validation Error", "Backlogs cannot be negative")
                return None
            if backlogs != int(backlogs):
                messagebox.showwarning("Warning", "Backlogs should be a whole number. Using rounded value.")
            features.append(float(int(backlogs)))  # Ensure it's an integer value
            
            # 10th aggregate
            tenth_agg = float(self.entries['tenth_agg'].get())
            if tenth_agg < 0 or tenth_agg > 100:
                messagebox.showerror("Validation Error", "10th aggregate must be between 0 and 100")
                return None
            features.append(tenth_agg)
            
            # 12th aggregate
            twelfth_agg = float(self.entries['twelfth_agg'].get())
            if twelfth_agg < 0 or twelfth_agg > 100:
                messagebox.showerror("Validation Error", "12th aggregate must be between 0 and 100")
                return None
            features.append(twelfth_agg)
            
            # Workshops
            workshops = float(self.entries['workshops'].get())
            if workshops < 0:
                messagebox.showerror("Validation Error", "Workshops cannot be negative")
                return None
            features.append(workshops)
            
            # Languages
            languages = float(self.entries['languages'].get())
            if languages < 0:
                messagebox.showerror("Validation Error", "Languages cannot be negative")
                return None
            features.append(languages)
            
            return features
            
        except ValueError as e:
            messagebox.showerror("Validation Error", f"Please enter valid numeric values: {e}")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Validation error: {e}")
            return None
    
    def predict_placement(self):
        """Handle prediction button click"""
        try:
            features = self.validate_inputs()
            if features is None:
                return
            
            prediction, confidence = self.predictor.predict(features)
            
            # Display result
            self.result_text.delete(1.0, tk.END)
            if prediction and prediction.strip():
                result = f"Predicted Placement: {prediction}\n"
                result += f"Confidence: {confidence:.2%}\n\n"
                result += f"Student Details:\n"
                result += f"Roll Number: {self.entries['roll_number'].get()}\n"
                result += f"Aggregate: {features[0]:.2f}%"
                self.result_text.insert(1.0, result)
                
                # Save to output file
                self.save_prediction(features, prediction)
                
                self.status_label.config(text=f"Prediction successful - Confidence: {confidence:.2%}")
            else:
                self.result_text.insert(1.0, "Student is not eligible for placement")
                self.status_label.config(text="Prediction: Not eligible")
                
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {e}")
            logger.error(f"Prediction error: {e}")
            self.status_label.config(text="Error occurred")
    
    def save_prediction(self, features: List[float], prediction: str):
        """Save prediction to output CSV file"""
        try:
            output_file = "b.csv"
            roll_number = self.entries['roll_number'].get()
            
            # Check if file exists and write header if needed
            file_exists = os.path.exists(output_file)
            
            with open(output_file, 'a', newline='') as f:
                if not file_exists:
                    f.write("Roll no.,Aggregate,10th_agg,12th_agg,Workshops,Languages,Prediction\n")
                
                line = f"{roll_number},{features[0]},{features[2]},{features[3]},{features[4]},{features[5]},{prediction}\n"
                f.write(line)
                
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
    
    def clear_fields(self):
        """Clear all input fields"""
        for key, widget in self.entries.items():
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.status_label.config(text="Fields cleared")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = PlacementGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

