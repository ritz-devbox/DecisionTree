"""
Placement Prediction System using Decision Tree Classifier
Improved version with better code structure, error handling, and best practices
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")


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
    
    def compare_models(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Compare multiple ML algorithms and return their performance
        
        Returns:
            Dictionary with model names and their metrics
        """
        try:
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Define models to compare
            models = {
                'Decision Tree': DecisionTreeClassifier(
                    max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
                )
            }
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
                )
            
            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    train_acc = model.score(X_train, y_train)
                    test_acc = model.score(X_test, y_test)
                    y_pred = model.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    results[name] = {
                        'model': model,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'classification_report': report
                    }
                    logger.info(f"{name} - Train: {train_acc:.2%}, Test: {test_acc:.2%}")
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    results[name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                return dict(zip(self.feature_names, importances))
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def get_recommendations(self, features: List[float]) -> List[Dict]:
        """
        Suggest improvements to increase placement probability
        
        Args:
            features: Current student features
            
        Returns:
            List of recommendations with expected improvements
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        try:
            current_pred, current_conf = self.predict(features)
            recommendations = []
            
            # Test different improvements
            test_cases = [
                ("Increase aggregate by 5%", [min(100, features[0] + 5)] + features[1:]),
                ("Reduce backlogs to 0", [features[0], 0] + features[2:]),
                ("Add 2 workshops", features[:4] + [features[4] + 2, features[5]]),
                ("Learn 1 more language", features[:5] + [min(5, features[5] + 1)]),
                ("Improve 10th aggregate by 5%", [features[0], features[1], 
                  min(100, features[2] + 5), features[3], features[4], features[5]]),
                ("Improve 12th aggregate by 5%", [features[0], features[1], features[2],
                  min(100, features[3] + 5), features[4], features[5]]),
            ]
            
            for desc, new_features in test_cases:
                try:
                    new_pred, new_conf = self.predict(new_features)
                    if new_conf > current_conf:
                        improvement = new_conf - current_conf
                        recommendations.append({
                            'action': desc,
                            'new_probability': new_conf,
                            'improvement': improvement,
                            'current_probability': current_conf
                        })
                except:
                    continue
            
            # Sort by improvement (descending)
            recommendations.sort(key=lambda x: x['improvement'], reverse=True)
            return recommendations[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict placement for multiple students from DataFrame
        
        Args:
            df: DataFrame with student features
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        try:
            predictions = []
            confidences = []
            
            for _, row in df.iterrows():
                try:
                    features = [
                        float(row.get('Aggregate', 0)),
                        float(row.get('Backlogs', 0)),
                        float(row.get('10th_agg', 0)),
                        float(row.get('12th_agg', 0)),
                        float(row.get('Workshops', 0)),
                        float(row.get('Languages', 0))
                    ]
                    pred, conf = self.predict(features)
                    predictions.append(pred)
                    confidences.append(conf)
                except Exception as e:
                    logger.warning(f"Error predicting for row {len(predictions)}: {e}")
                    predictions.append("Error")
                    confidences.append(0.0)
            
            result_df = df.copy()
            result_df['Prediction'] = predictions
            result_df['Confidence'] = confidences
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise


class PredictionHistory:
    """Manage prediction history"""
    
    def __init__(self, history_file: str = "prediction_history.json"):
        self.history_file = history_file
        self.history = self.load_history()
    
    def load_history(self) -> List[Dict]:
        """Load prediction history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_prediction(self, features: List[float], prediction: str, confidence: float, roll_number: str = ""):
        """Save a prediction to history"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'roll_number': roll_number,
            'features': features,
            'prediction': prediction,
            'confidence': confidence
        }
        self.history.append(entry)
        self.save_history()
    
    def save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get recent predictions"""
        return self.history[-n:] if len(self.history) > n else self.history
    
    def clear_history(self):
        """Clear all history"""
        self.history = []
        self.save_history()


class PlacementGUI:
    """GUI application for placement prediction"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Training and Placement Cell - Placement Prediction System')
        self.root.geometry('900x800')
        self.root.configure(bg='lightblue')
        
        self.predictor = PlacementPredictor()
        self.history = PredictionHistory()
        self.model_type = tk.StringVar(value="Decision Tree")
        self.model_comparison_results = None
        
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
        
        # Model selection
        model_label = tk.Label(main_frame, text='Model:', bg="lightblue", font=("Arial", 10, "bold"))
        model_label.grid(row=len(fields), column=0, sticky='w', pady=5, padx=10)
        
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_type, 
                                   values=["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"],
                                   state="readonly", width=27)
        model_combo.grid(row=len(fields), column=1, pady=5, padx=10, sticky='w')
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='lightblue')
        button_frame.grid(row=len(fields)+1, column=0, columnspan=2, pady=20)
        
        submit_btn = tk.Button(
            button_frame,
            text='Predict Placement',
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.predict_placement,
            padx=15,
            pady=8
        )
        submit_btn.pack(side='left', padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text='Clear',
            bg="#f44336",
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.clear_fields,
            padx=15,
            pady=8
        )
        clear_btn.pack(side='left', padx=5)
        
        # Advanced features buttons frame
        advanced_frame = tk.Frame(main_frame, bg='lightblue')
        advanced_frame.grid(row=len(fields)+2, column=0, columnspan=2, pady=10)
        
        batch_btn = tk.Button(
            advanced_frame,
            text='Batch Predict',
            bg="#2196F3",
            fg="white",
            font=("Arial", 10),
            command=self.batch_predict,
            padx=10,
            pady=5
        )
        batch_btn.pack(side='left', padx=3)
        
        recommend_btn = tk.Button(
            advanced_frame,
            text='Get Recommendations',
            bg="#FF9800",
            fg="white",
            font=("Arial", 10),
            command=self.show_recommendations,
            padx=10,
            pady=5
        )
        recommend_btn.pack(side='left', padx=3)
        
        feature_btn = tk.Button(
            advanced_frame,
            text='Feature Importance',
            bg="#9C27B0",
            fg="white",
            font=("Arial", 10),
            command=self.show_feature_importance,
            padx=10,
            pady=5
        )
        feature_btn.pack(side='left', padx=3)
        
        compare_btn = tk.Button(
            advanced_frame,
            text='Compare Models',
            bg="#607D8B",
            fg="white",
            font=("Arial", 10),
            command=self.compare_models,
            padx=10,
            pady=5
        )
        compare_btn.pack(side='left', padx=3)
        
        history_btn = tk.Button(
            advanced_frame,
            text='View History',
            bg="#795548",
            fg="white",
            font=("Arial", 10),
            command=self.show_history,
            padx=10,
            pady=5
        )
        history_btn.pack(side='left', padx=3)
        
        # Result area
        result_label = tk.Label(main_frame, text='Prediction Result:', bg="lightblue", font=("Arial", 12, "bold"))
        result_label.grid(row=len(fields)+3, column=0, columnspan=2, pady=(20, 5), sticky='w', padx=10)
        
        self.result_text = tk.Text(
            main_frame,
            width=60,
            height=6,
            bg="lightyellow",
            wrap=tk.WORD,
            font=("Arial", 10)
        )
        self.result_text.grid(row=len(fields)+4, column=0, columnspan=2, pady=5, padx=10, sticky='w')
        
        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text='Ready',
            bg="lightblue",
            font=("Arial", 9),
            anchor='w'
        )
        self.status_label.grid(row=len(fields)+5, column=0, columnspan=2, pady=10, padx=10, sticky='w')
    
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
                
                # Save to history
                roll_number = self.entries['roll_number'].get()
                self.history.save_prediction(features, prediction, confidence, roll_number)
                
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
    
    def on_model_change(self, event=None):
        """Handle model selection change"""
        selected = self.model_type.get()
        self.status_label.config(text=f"Selected model: {selected}. Train model to use it.")
    
    def batch_predict(self):
        """Batch prediction from CSV file"""
        from tkinter import filedialog
        try:
            file_path = filedialog.askopenfilename(
                title="Select CSV file with student data",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not file_path:
                return
            
            df = pd.read_csv(file_path)
            result_df = self.predictor.batch_predict(df)
            
            # Save results
            output_path = filedialog.asksaveasfilename(
                title="Save predictions",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if output_path:
                result_df.to_csv(output_path, index=False)
                messagebox.showinfo("Success", 
                    f"Batch prediction complete!\n{len(result_df)} predictions saved to:\n{output_path}")
                self.status_label.config(text=f"Batch prediction: {len(result_df)} students processed")
        except Exception as e:
            messagebox.showerror("Error", f"Batch prediction failed: {e}")
            logger.error(f"Batch prediction error: {e}")
    
    def show_recommendations(self):
        """Show improvement recommendations"""
        try:
            features = self.validate_inputs()
            if features is None:
                return
            
            recommendations = self.predictor.get_recommendations(features)
            
            if not recommendations:
                messagebox.showinfo("Recommendations", 
                    "No specific recommendations. Your profile looks good!")
                return
            
            # Create recommendations window
            rec_window = tk.Toplevel(self.root)
            rec_window.title("Improvement Recommendations")
            rec_window.geometry("500x400")
            
            text_widget = tk.Text(rec_window, wrap=tk.WORD, padx=10, pady=10, font=("Arial", 10))
            text_widget.pack(fill='both', expand=True)
            
            current_pred, current_conf = self.predictor.predict(features)
            text_widget.insert('1.0', f"Current Placement Probability: {current_conf:.1%}\n\n")
            text_widget.insert('end', "Recommendations to improve:\n" + "="*50 + "\n\n")
            
            for i, rec in enumerate(recommendations, 1):
                text_widget.insert('end', 
                    f"{i}. {rec['action']}\n"
                    f"   New Probability: {rec['new_probability']:.1%} "
                    f"(+{rec['improvement']:.1%})\n\n")
            
            text_widget.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error getting recommendations: {e}")
    
    def show_feature_importance(self):
        """Display feature importance"""
        try:
            if self.predictor.model is None:
                messagebox.showwarning("Warning", "Please train a model first.")
                return
            
            importance = self.predictor.get_feature_importance()
            if not importance:
                messagebox.showinfo("Info", "Feature importance not available for this model.")
                return
            
            # Create window
            imp_window = tk.Toplevel(self.root)
            imp_window.title("Feature Importance")
            imp_window.geometry("400x300")
            
            text_widget = tk.Text(imp_window, wrap=tk.WORD, padx=10, pady=10, font=("Arial", 10))
            text_widget.pack(fill='both', expand=True)
            
            text_widget.insert('1.0', "Feature Importance Analysis\n" + "="*40 + "\n\n")
            
            # Sort by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, score in sorted_features:
                text_widget.insert('end', f"{feature}: {score:.3f}\n")
            
            text_widget.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error showing feature importance: {e}")
    
    def compare_models(self):
        """Compare different ML models"""
        try:
            self.status_label.config(text="Comparing models... This may take a moment.")
            self.root.update()
            
            results = self.predictor.compare_models()
            
            # Create comparison window
            comp_window = tk.Toplevel(self.root)
            comp_window.title("Model Comparison")
            comp_window.geometry("600x400")
            
            text_widget = tk.Text(comp_window, wrap=tk.WORD, padx=10, pady=10, font=("Arial", 10))
            text_widget.pack(fill='both', expand=True)
            
            text_widget.insert('1.0', "Model Performance Comparison\n" + "="*50 + "\n\n")
            
            for name, result in results.items():
                if 'error' in result:
                    text_widget.insert('end', f"{name}: Error - {result['error']}\n\n")
                else:
                    text_widget.insert('end', 
                        f"{name}:\n"
                        f"  Train Accuracy: {result['train_accuracy']:.2%}\n"
                        f"  Test Accuracy: {result['test_accuracy']:.2%}\n\n")
            
            # Find best model
            best_model = max([(k, v) for k, v in results.items() if 'test_accuracy' in v], 
                           key=lambda x: x[1]['test_accuracy'], default=None)
            if best_model:
                text_widget.insert('end', f"\nBest Model: {best_model[0]} "
                                        f"({best_model[1]['test_accuracy']:.2%} test accuracy)")
            
            text_widget.config(state='disabled')
            self.status_label.config(text="Model comparison complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error comparing models: {e}")
            self.status_label.config(text="Error in model comparison")
    
    def show_history(self):
        """Show prediction history"""
        try:
            recent = self.history.get_recent(20)
            
            if not recent:
                messagebox.showinfo("History", "No prediction history available.")
                return
            
            # Create history window
            hist_window = tk.Toplevel(self.root)
            hist_window.title("Prediction History")
            hist_window.geometry("700x500")
            
            # Create scrollable text widget
            frame = tk.Frame(hist_window)
            frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side='right', fill='y')
            
            text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, 
                                 font=("Arial", 9))
            text_widget.pack(fill='both', expand=True)
            scrollbar.config(command=text_widget.yview)
            
            text_widget.insert('1.0', f"Prediction History ({len(recent)} recent entries)\n" + "="*70 + "\n\n")
            
            for entry in reversed(recent):  # Show newest first
                timestamp = entry.get('timestamp', 'Unknown')
                roll = entry.get('roll_number', 'N/A')
                pred = entry.get('prediction', 'N/A')
                conf = entry.get('confidence', 0)
                features = entry.get('features', [])
                
                # Safely access features with defaults if list is incomplete
                if len(features) >= 6:
                    features_str = (f"Features: Aggregate={features[0]:.1f}%, Backlogs={features[1]}, "
                                   f"10th={features[2]:.1f}%, 12th={features[3]:.1f}%, "
                                   f"Workshops={features[4]}, Languages={features[5]}")
                else:
                    # Handle incomplete or missing features
                    feature_defaults = [0.0, 0, 0.0, 0.0, 0, 0]
                    for i in range(min(len(features), 6)):
                        feature_defaults[i] = features[i]
                    features_str = (f"Features: Aggregate={feature_defaults[0]:.1f}%, Backlogs={feature_defaults[1]}, "
                                   f"10th={feature_defaults[2]:.1f}%, 12th={feature_defaults[3]:.1f}%, "
                                   f"Workshops={feature_defaults[4]}, Languages={feature_defaults[5]}")
                    if len(features) < 6:
                        features_str += " (incomplete data)"
                
                text_widget.insert('end', 
                    f"Time: {timestamp}\n"
                    f"Roll Number: {roll}\n"
                    f"Prediction: {pred} (Confidence: {conf:.1%})\n"
                    f"{features_str}\n"
                    f"{'-'*70}\n\n")
            
            text_widget.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error showing history: {e}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = PlacementGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

