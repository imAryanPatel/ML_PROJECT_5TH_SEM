# 🌾 Crop Recommendation System

A Machine Learning-powered web application that recommends the best crop to grow based on soil nutrients and environmental conditions.
#Link:  https://cropyieldprediction-hvk.streamlit.app

## 📋 Features

- Interactive web interface built with Streamlit
- Random Forest Classifier with hyperparameter tuning
- Real-time crop recommendations
- Top 3 crop suggestions with confidence scores
- Input validation and user-friendly design

## 🗂️ Folder Structure

```
crop-recommendation-app/
│
├── app.py                          # Streamlit web application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/
│   └── Crop_recommendation.csv    # Dataset
│
└── models/
    ├── best_random_forest_model.pkl  # Trained model
    ├── scaler.pkl                     # Feature scaler
    └── label_encoder.pkl              # Label encoder
```

## 🚀 Installation & Setup

### Step 1: Clone or Create Project Directory

```bash
mkdir crop-recommendation-app
cd crop-recommendation-app
```

### Step 2: Create Required Folders

```bash
mkdir data models
```

### Step 3: Add Files

Place the following files in the appropriate directories:
- `Crop_recommendation.csv` → in `data/` folder
- `app.py` → in root directory
- `train_model.py` → in root directory
- `requirements.txt` → in root directory

### Step 4: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Train the Model

First, train the model to generate the pickle files:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Handle outliers
- Train a Random Forest model with hyperparameter tuning
- Save the model, scaler, and label encoder in the `models/` folder

Expected output:
```
Best model - Test Accuracy: 0.99XX
✓ Saved: models/best_random_forest_model.pkl
✓ Saved: models/scaler.pkl
✓ Saved: models/label_encoder.pkl
```

### Run the Streamlit App

After training, launch the web application:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 How to Use the Web App

1. **Enter Soil Nutrients:**
   - Nitrogen (N) ratio
   - Phosphorus (P) ratio
   - Potassium (K) ratio
   - pH value

2. **Enter Climate Conditions:**
   - Temperature (°C)
   - Humidity (%)
   - Rainfall (mm)

3. **Click "Get Crop Recommendation"**

4. **View Results:**
   - Recommended crop with confidence score
   - Top 3 crop suggestions
   - Input summary table

## 🔧 Model Details

- **Algorithm:** Random Forest Classifier
- **Features:** 7 input parameters (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Preprocessing:** StandardScaler for feature scaling, Outlier handling using IQR method
- **Hyperparameters Tuned:**
  - n_estimators: [100, 200, 300]
  - max_depth: [None, 10, 20]
  - min_samples_split: [2, 5, 10]

## 📈 Model Performance

The model typically achieves:
- Training Accuracy: ~99%
- Test Accuracy: ~99%

## 🛠️ Troubleshooting

### Model files not found
If you see "Model files not found!", make sure you've run `train_model.py` first to generate the pickle files.

### Module not found errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CSV file not found
Make sure `Crop_recommendation.csv` is in the `data/` folder.

## 📦 Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- seaborn
- matplotlib
- Pillow

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created with ❤️ for precision agriculture and smart farming

---

**Happy Farming! 🌱**


