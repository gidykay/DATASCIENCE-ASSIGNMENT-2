# DATASCIENCE-ASSIGNMENT-2
# Road Accident Severity Analysis

## Description
This project analyzes road accident severity using a linear regression model.

## Files
- **prac.py**: Main script to load data, train model, evaluate, and save the model.
- **Road.csv**: The dataset used for training and testing.
- **road_accident_model.pkl**: Saved model for future predictions.
- **results.html**: Contains evaluation metrics and discussion of results.
- **actual_vs_predicted.png**, **predicted_severity_distribution.png**, **residual_plot.png**: Visualizations.
- **README.md**: Overview and instructions.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (if provided).
3. Run `python prac.py`.
4. Check the output metrics and generated plots.

## Example Usage
After training, you can load the model and predict severity for a new set of conditions by running:
```python
with open('road_accident_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Construct a DataFrame for new conditions
...
# model.predict(new_data)
