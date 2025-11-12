# How to Run Your Decision Tree Classifier Code

## 📋 **Step 1: Install Required Packages**

Open Command Prompt or PowerShell in your project directory and run:

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install pandas numpy scikit-learn joblib
```

**Note:** `tkinter` comes with Python by default. If you get an error, you may need:
- **Windows:** Usually included
- **Linux:** `sudo apt-get install python3-tk`
- **Mac:** Usually included

---

## 🚀 **Step 2: Prepare Required Files**

Make sure you have the training data file in the same folder as `DecisionTree.py`:

### **`a.csv`** - Your training data
The CSV file should have the following structure:
- First column: Roll Number or ID (will be skipped)
- Columns 1-6: Features (Aggregate, Backlogs, 10th_agg, 12th_agg, Workshops, Languages)
- Last column: Target (Placement prediction - e.g., "Placed", "Not Placed", or company name)

Example format:
```csv
RollNo,Aggregate,Backlogs,10th_agg,12th_agg,Workshops,Languages,Target
1234567,75,0,85,80,3,2,Placed
1234568,65,1,75,70,2,1,Not Placed
1234569,80,0,90,85,4,3,Placed
```

---

## 🎯 **Step 3: Run the Code**

### **Option A: Command Line (Recommended)**

1. Open Command Prompt or PowerShell
2. Navigate to your project folder:
   ```bash
   cd C:\Users\nedun\source\repos\DecisionTree
   ```
3. Run the script:
   ```bash
   python DecisionTree.py
   ```

### **Option B: From IDE (VS Code, PyCharm, etc.)**

1. Open `DecisionTree.py` in your IDE
2. Press **F5** or click the Run button
3. Or use the terminal in your IDE and run: `python DecisionTree.py`

### **Option C: Double-click (Windows)**

1. Right-click `DecisionTree.py`
2. Select "Open with" → "Python"

---

## 🔧 **Troubleshooting**

### **Error: "No module named 'pandas'", "No module named 'sklearn'", etc.**
```bash
pip install -r requirements.txt
```

### **Error: "FileNotFoundError: a.csv"**
- Make sure `a.csv` is in the same folder as `DecisionTree.py`
- The file should contain training data with the correct format (see Step 2)

### **Error: "TclError" (tkinter)**
- **Windows:** Usually works out of the box
- **Linux:** `sudo apt-get install python3-tk`
- **Mac:** Usually included

### **Error: "Model not trained or loaded"**
- Make sure `a.csv` exists and has valid data
- The program will automatically train a model on first run
- A trained model will be saved as `placement_model.pkl` for future use

---

## 📊 **How to Use the GUI**

1. **Enter Student Details:**
   - **Roll Number** (7-8 digits, optional)
   - **Aggregate (%)** - Overall percentage (0-100, warning if < 50)
   - **Backlogs** - Select 0 or 1 using radio buttons
   - **10th Aggregate (%)** - 10th grade percentage (0-100)
   - **12th Aggregate (%)** - 12th grade percentage (0-100)
   - **Workshops** - Number of workshops attended (≥ 0)
   - **Programming Languages** - Number of languages known (≥ 0)

2. **Click "Predict Placement"** to get prediction

3. **View Result** in the yellow text box showing:
   - Predicted placement status
   - Confidence level
   - Student details

4. **Click "Clear"** to reset all fields

5. **Results are automatically saved** to `b.csv` file

---

## ✅ **Quick Test**

Create a minimal `a.csv` to test:

```csv
RollNo,Aggregate,Backlogs,10th_agg,12th_agg,Workshops,Languages,Target
1234567,75,0,85,80,3,2,Placed
1234568,65,1,75,70,2,1,Not Placed
1234569,80,0,90,85,4,3,Placed
1234570,55,0,65,60,1,1,Not Placed
```

Then run: `python DecisionTree.py`

---

## 🚀 **Expected Behavior**

1. **First Run:**
   - GUI window opens (800x700 pixels, light blue background)
   - Model trains automatically using `a.csv`
   - Model is saved as `placement_model.pkl`
   - You see input fields for student data

2. **Subsequent Runs:**
   - GUI window opens
   - Pre-trained model loads from `placement_model.pkl` (faster startup)
   - You can immediately make predictions

3. **Making Predictions:**
   - Enter student data in the form
   - Click "Predict Placement"
   - Prediction appears in the yellow text box
   - Data is saved to `b.csv`

---

## 📁 **Generated Files**

The program will create/update these files:
- **`placement_model.pkl`** - Trained model (created after first run)
- **`b.csv`** - Prediction results (appended with each prediction)

---

That's it! The program should run and open a GUI window for placement predictions.

