# California House Price Prediction

A comprehensive end-to-end machine learning project for predicting house prices in California districts using the California Housing dataset.

## 📊 Project Overview

This project demonstrates a complete machine learning workflow including data acquisition, exploratory data analysis, feature engineering, model training, and evaluation. The goal is to predict median house values in California districts based on various housing and demographic features.

### Key Features
- **End-to-End ML Pipeline**: Complete workflow from data loading to model deployment
- **Advanced Feature Engineering**: Custom transformers and feature combinations
- **Stratified Sampling**: Ensures representative train/test splits
- **Comprehensive EDA**: Detailed data visualization and insights
- **Production-Ready Code**: Modular functions and pipelines for reusability

## 🎯 Problem Statement

Predict the median house value for California districts given various features such as:
- Location (longitude, latitude)
- Housing characteristics (age, rooms, bedrooms)
- Demographics (population, households, income)
- Proximity to ocean

## 📁 Project Structure

```
Cali_house_price_prediction/
├── House_Price_E2E.ipynb              # Original notebook
├── House_Price_E2E_Commented.ipynb    # Well-commented version
├── README.md                          # This file
├── datasets/                          # Data directory (created automatically)
│   └── housing/
│       ├── housing.csv               # Main dataset
│       └── housing.tgz              # Compressed dataset
└── requirements.txt                   # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Cali_house_price_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook House_Price_E2E_Commented.ipynb
   ```

## 📈 Dataset Information

### Source
- **Dataset**: California Housing Dataset
- **Source**: [Hands-On Machine Learning with Scikit-Learn & TensorFlow](https://github.com/ageron/handson-ml2)
- **Size**: 20,640 samples, 9 features

### Features
| Feature | Type | Description | Missing Values |
|---------|------|-------------|----------------|
| longitude | float64 | Geographic longitude | 0 |
| latitude | float64 | Geographic latitude | 0 |
| housing_median_age | float64 | Median age of houses | 0 |
| total_rooms | float64 | Total number of rooms | 0 |
| total_bedrooms | float64 | Total number of bedrooms | 207 |
| population | float64 | Total population | 0 |
| households | float64 | Total number of households | 0 |
| median_income | float64 | Median income | 0 |
| ocean_proximity | object | Proximity to ocean | 0 |

### Target Variable
- **median_house_value**: Median house value in USD (target for prediction)

## 🔍 Key Insights from EDA

### Geographic Patterns
- Housing prices are higher near the ocean, especially in southern coastal areas
- Northern coastal regions don't follow the same price pattern
- Clear correlation between location and house values

### Feature Correlations
- **Strongest correlation**: `median_income` (0.69)
- **Moderate correlations**: `total_rooms`, `housing_median_age`
- **Weak correlations**: `population`, `households`

### Derived Features
- `rooms_per_household`: Better predictor than raw `total_rooms`
- `bedrooms_per_room`: Useful for understanding house layout
- `population_per_household`: Demographic density indicator

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline

```python
# Complete preprocessing pipeline
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                          "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
], remainder=default_num_pipeline)
```

### Key Components

1. **Missing Value Handling**: Median imputation for numerical features
2. **Categorical Encoding**: One-hot encoding for `ocean_proximity`
3. **Feature Scaling**: StandardScaler for numerical features
4. **Feature Engineering**: 
   - Ratio features (bedrooms/rooms, rooms/household)
   - Log transformations for skewed features
   - Geographic clustering for location-based features

### Model Performance

| Model | RMSE | Notes |
|-------|------|-------|
| Linear Regression | ~$68,628 | Baseline model |
| (Future: Random Forest) | TBD | Expected improvement |
| (Future: XGBoost) | TBD | Expected best performance |

## 📊 Visualization Examples

The project includes comprehensive visualizations:

1. **Geographic Distribution**: Scatter plots showing house locations and prices
2. **Feature Distributions**: Histograms for all numerical features
3. **Correlation Analysis**: Correlation matrix and scatter plots
4. **Price Patterns**: Color-coded maps showing price variations

## 🔧 Custom Components

### ClusterSimilarity Transformer
```python
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Creates similarity features based on geographic clustering.
    Uses RBF kernel to measure similarity to cluster centers.
    """
```

### Feature Engineering Functions
- `column_ratio()`: Creates ratio features
- `ratio_pipeline()`: Complete pipeline for ratio features
- `log_pipeline()`: Pipeline for log transformations

## 📋 Best Practices Implemented

### Data Splitting
- **Stratified Sampling**: Based on income categories to ensure representative splits
- **No Data Leakage**: Strict separation of train/test sets
- **Reproducibility**: Fixed random seeds for consistent results

### Code Organization
- **Modular Functions**: Reusable transformation functions
- **Pipeline Approach**: Sklearn pipelines for production-ready code
- **Documentation**: Comprehensive comments and docstrings

### Error Handling
- **Missing Values**: Proper imputation strategies
- **Unknown Categories**: Handle unknown values in categorical features
- **Compatibility**: Monkey patching for sklearn version compatibility

## 🚀 Future Enhancements

### Model Improvements
- [ ] Implement Random Forest regression
- [ ] Add XGBoost and LightGBM models
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Ensemble methods (voting, stacking)

### Feature Engineering
- [ ] More sophisticated geographic features
- [ ] Time-based features (if temporal data available)
- [ ] Interaction features between key variables
- [ ] Polynomial features for non-linear relationships

### Production Deployment
- [ ] API development with FastAPI
- [ ] Model serialization and versioning
- [ ] Automated retraining pipeline
- [ ] Monitoring and logging

## 📚 Dependencies

Create a `requirements.txt` file with:

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset Source**: Aurélien Géron's "Hands-On Machine Learning with Scikit-Learn & TensorFlow"
- **Inspiration**: End-to-end machine learning project structure
- **Tools**: Scikit-learn, Pandas, Matplotlib, Jupyter

## 📞 Contact

For questions or suggestions, please open an issue in the repository or contact the maintainer.

---

**Note**: This is an educational project demonstrating best practices in machine learning. For production use, additional considerations like model validation, monitoring, and security should be implemented. 