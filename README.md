# Student Placement Prediction System

A comprehensive machine learning application for predicting student placement outcomes using multiple ML algorithms. Built with Python, Tkinter, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Screenshots](#-screenshots)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Features in Detail](#-features-in-detail)
- [Dataset](#-dataset)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### Core Features
- 🎯 **Placement Prediction** - Predict student placement probability using ML models
- 🤖 **Multiple ML Algorithms** - Support for Decision Tree, Random Forest, Gradient Boosting, and XGBoost
- 📊 **Feature Importance Analysis** - Understand which features impact placement most
- 💡 **Recommendation System** - Get personalized suggestions to improve placement chances
- 📦 **Batch Prediction** - Process multiple students from CSV files
- 📈 **Model Comparison** - Compare performance of different ML algorithms
- 📝 **Prediction History** - Track all predictions with timestamps
- 💾 **Model Persistence** - Save and load trained models
- 🎨 **User-Friendly GUI** - Intuitive desktop application built with Tkinter

### Advanced Features
- ✅ Input validation and error handling
- ✅ Real-time confidence scores
- ✅ Export predictions to CSV
- ✅ Comprehensive logging
- ✅ Model performance metrics

---

## 🖼️ Screenshots

### Main Interface
The application provides a clean, intuitive interface for entering student data and viewing predictions.

### Key Features
- **Model Selection** - Choose from multiple ML algorithms
- **Batch Processing** - Upload CSV files for bulk predictions
- **Recommendations** - Get actionable improvement suggestions
- **History Tracking** - View all past predictions

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/ritz-devbox/DecisionTree.git
cd DecisionTree
```

### Step 2: Create a virtual Environment & Install Dependencies
```bash
python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt
```

### Optional: Install XGBoost (for XGBoost model support)
```bash
pip install xgboost
```

### Step 3: Prepare Data
Ensure you have a training dataset file named `a.csv` in the project directory. The file should have the following structure:

```csv
RollNo,Aggregate,Backlogs,10th_agg,12th_agg,Workshops,Languages,Target
1234567,75,0,85,80,3,2,Placed
1234568,65,1,75,70,2,1,Not Placed
...
```

**Note:** If you don't have `a.csv`, you can generate a sample dataset using:
```bash
python generate_dataset.py
```

---

## ⚡ Quick Start

1. **Run the application:**
   ```bash
   python DecisionTree.py
   ```

2. **Enter student data:**
   - Roll Number
   - Aggregate (%)
   - Backlogs
   - 10th Aggregate (%)
   - 12th Aggregate (%)
   - Workshops
   - Programming Languages

3. **Click "Predict Placement"** to get the prediction

4. **Explore advanced features:**
   - Click "Get Recommendations" for improvement suggestions
   - Click "Feature Importance" to see which features matter most
   - Click "Compare Models" to see algorithm performance
   - Click "Batch Predict" to process multiple students

---

## 📖 Usage

### Basic Prediction

1. Launch the application
2. Fill in the student information fields
3. Select a model from the dropdown (optional)
4. Click **"Predict Placement"**
5. View the prediction and confidence score in the result area

### Batch Prediction

1. Click **"Batch Predict"** button
2. Select a CSV file containing student data
3. The application will process all students
4. Save the results to a new CSV file

### Get Recommendations

1. Enter student data
2. Click **"Get Recommendations"**
3. View personalized suggestions to improve placement probability

### Compare Models

1. Click **"Compare Models"**
2. Wait for the comparison to complete (may take a moment)
3. View performance metrics for each algorithm
4. See which model performs best

### View History

1. Click **"View History"**
2. Browse through past predictions
3. See timestamps, roll numbers, and confidence scores

---

## 📁 Project Structure

```
DecisionTree/
│
├── DecisionTree.py          # Main application file
├── requirements.txt         # Python dependencies
├── generate_dataset.py      # Dataset generation script
├── a.csv                    # Training dataset (input)
├── b.csv                    # Prediction results (output)
├── placement_model.pkl      # Saved trained model
├── prediction_history.json  # Prediction history
│
├── README.md                # This file
├── HOW_TO_RUN_DECISION_TREE.md      # Detailed run instructions
├── HOW_TO_RUN_AND_VERIFY.md          # Verification guide
├── DATASET_IMPROVEMENTS.md           # Dataset benefits
├── DECISION_TREE_IMPROVEMENTS.md     # Code improvements
├── PROJECT_EXPANSION_ROADMAP.md      # Future enhancements
└── QUICK_WINS_IMPLEMENTATION.md      # Quick implementation guide
```

---

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **Tkinter** - GUI framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **joblib** - Model serialization
- **XGBoost** (optional) - Gradient boosting framework

---

## 🎯 Features in Detail

### 1. Multiple ML Algorithms
The application supports multiple machine learning algorithms:
- **Decision Tree** - Fast, interpretable
- **Random Forest** - Ensemble method, robust
- **Gradient Boosting** - High performance
- **XGBoost** - State-of-the-art (optional)

### 2. Feature Importance
Visualize which student attributes have the most impact on placement:
- Aggregate percentage
- Backlogs
- 10th and 12th aggregate
- Workshops attended
- Programming languages known

### 3. Recommendation System
Get personalized suggestions:
- Increase aggregate by X%
- Reduce backlogs
- Attend more workshops
- Learn additional languages
- Improve 10th/12th scores

### 4. Batch Processing
Process multiple students at once:
- Upload CSV file
- Automatic validation
- Bulk predictions
- Export results

### 5. Model Comparison
Compare algorithm performance:
- Train accuracy
- Test accuracy
- Classification reports
- Best model identification

---

## 📊 Dataset

The application uses a training dataset (`a.csv`) with the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| RollNo | Student roll number | 7-8 digits |
| Aggregate | Overall percentage | 0-100% |
| Backlogs | Number of backlogs | 0+ |
| 10th_agg | 10th grade percentage | 0-100% |
| 12th_agg | 12th grade percentage | 0-100% |
| Workshops | Workshops attended | 0-5 |
| Languages | Programming languages known | 1-5 |
| Target | Placement outcome | Placed/Not Placed |

**Current Dataset:** 1,200+ student records with realistic patterns and correlations.

---

## 🔮 Future Enhancements

See [PROJECT_EXPANSION_ROADMAP.md](PROJECT_EXPANSION_ROADMAP.md) for detailed expansion plans:

- 🌐 Web interface (Flask/FastAPI)
- 🗄️ Database integration
- 🔐 User authentication
- 📈 Advanced analytics dashboard
- 🔄 Automated model retraining
- 📱 Mobile app support
- 🌍 Multi-language support
- 🔗 API endpoints

---

## 🐛 Troubleshooting

### Common Issues

**Error: "No module named 'sklearn'"**
```bash
pip install scikit-learn
```

**Error: "FileNotFoundError: a.csv"**
- Ensure `a.csv` exists in the project directory
- Or run `python generate_dataset.py` to create a sample dataset

**GUI doesn't open**
- On Linux: `sudo apt-get install python3-tk`
- On Windows/Mac: Usually included with Python

**XGBoost warning**
- This is normal if XGBoost is not installed
- Install with: `pip install xgboost` (optional)

---

## 📝 Example Usage

### Example 1: Single Prediction
```
Input:
- Aggregate: 85%
- Backlogs: 0
- 10th Aggregate: 90%
- 12th Aggregate: 88%
- Workshops: 4
- Languages: 3

Output:
- Prediction: Placed
- Confidence: 92.5%
```

### Example 2: Batch Prediction
```python
# CSV file format:
RollNo,Aggregate,Backlogs,10th_agg,12th_agg,Workshops,Languages
1234567,75,0,85,80,3,2
1234568,65,1,75,70,2,1

# Results saved with predictions and confidence scores
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Ritish Nedunoori** - *Initial work*

---

## 🙏 Acknowledgments

- scikit-learn community for excellent ML tools
- Python community for amazing libraries
- All contributors and users

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the documentation files in the repository
- Review [HOW_TO_RUN_DECISION_TREE.md](HOW_TO_RUN_DECISION_TREE.md) for detailed instructions

---

## ⭐ Show Your Support

If you find this project helpful, please give it a star! ⭐

---

**Made with ❤️ for students and placement cells**

---

## 📚 Additional Documentation

- [How to Run](HOW_TO_RUN_DECISION_TREE.md) - Detailed run instructions
- [Verification Guide](HOW_TO_RUN_AND_VERIFY.md) - Testing and verification
- [Dataset Benefits](DATASET_IMPROVEMENTS.md) - Why large datasets matter
- [Code Improvements](DECISION_TREE_IMPROVEMENTS.md) - Technical improvements
- [Expansion Roadmap](PROJECT_EXPANSION_ROADMAP.md) - Future enhancements
- [Quick Wins](QUICK_WINS_IMPLEMENTATION.md) - Easy implementations

---

*Last Updated: 2025*

