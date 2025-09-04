# NBA Statistic Analysis Project

## Overview
This project is a comprehensive NBA player analysis and ranking system that uses machine learning algorithms to predict player rankings based on various statistical metrics. The system analyzes player performance data from multiple NBA seasons and provides tools for player evaluation, ranking predictions, and trade analysis.In pdf CSCI 184 Final Project Report have an overview process of the entire project.

## Features

### 🏀 Player Ranking System
- **Machine Learning Models**: Implements K-Nearest Neighbors (KNN) and Decision Tree algorithms for player ranking predictions
- **Statistical Analysis**: Analyzes comprehensive player statistics including scoring, rebounding, assists, shooting percentages, and advanced metrics
- **Performance Metrics**: Evaluates model accuracy using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² scores

### 📊 Data Analysis Tools
- **Multi-Season Analysis**: Processes NBA data from 2022, 2024, and current seasons
- **Statistical Visualization**: Creates charts and graphs for data exploration and analysis
- **Player Comparison**: Enables side-by-side comparison of player statistics and rankings

### 🔄 Trade Simulator
- **Trade Evaluation**: Calculates trade values based on player rankings
- **Player Labeling**: Categorizes players into performance tiers (Elite, All-Star, Starter, Role Player, Bench)
- **Trade Balance Assessment**: Determines which team benefits more from a proposed trade

## Datasets

The project works with several NBA datasets:

- **`NBA2024.csv`**: Current season player statistics and rankings
- **`nba2022.csv`**: 2022 season data for training machine learning models
- **`CurrentMVP.csv`**: MVP candidate statistics including advanced metrics like Win Shares and VORP
- **`nba_statistic_2024_rank.csv`**: Ranked player statistics for the 2024-25 season

## Key Statistics Analyzed

### Basic Statistics
- **Scoring**: Points Per Game (PPG), Field Goal Percentage (FG%), 3-Point Percentage (3P%)
- **Rebounding**: Rebounds Per Game (RPG), Offensive/Defensive Rebounds
- **Playmaking**: Assists Per Game (APG), Turnovers Per Game (TPG)
- **Defense**: Steals Per Game (SPG), Blocks Per Game (BPG)

### Advanced Metrics
- **Efficiency**: Effective Field Goal Percentage (eFG%), True Shooting Percentage (TS%)
- **Impact**: Win Shares (WS), Value Over Replacement Player (VORP), Box Plus/Minus (BPM)
- **Usage**: Usage Rate (USG%), Minutes Per Game (MPG)

## Machine Learning Models

### 1. Decision Tree Regressor
- **Purpose**: Predicts player rankings based on statistical features
- **Features**: MPG, TO%, FTA, FT%, 2PA, 2P%, 3PA, 3P%, PPG, RPG, APG, SPG, BPG, TPG
- **Output**: Predicted player rankings

### 2. K-Nearest Neighbors (KNN)
- **Purpose**: Alternative ranking prediction model
- **Implementation**: Uses KNN regressor for ranking predictions
- **Comparison**: Evaluated against actual rankings for accuracy

## Project Structure

```
NBA Statistic Analysis - 184 Proj/
├── 184Final.ipynb              # Main analysis notebook
├── 184Final-Copy1.ipynb        # Backup copy of main notebook
├── NBA2024.csv                 # Current season data
├── nba2022.csv                 # 2022 season data
├── CurrentMVP.csv              # MVP candidate data
├── nba_statistic_2024_rank.csv # Ranked 2024-25 season data
├── predicted_rank.csv          # Model predictions output
├── knn_ranked.csv             # KNN model results
├── player_ranks.csv           # Player rankings with labels
├── player_ranks_with_labels.csv # Labeled player rankings
├── modified_names.csv          # Processed player names
├── modified_rankings.csv       # Modified ranking data
└── name_order_mismatches.csv   # Name matching issues
```

## Usage

### Running the Analysis
1. Open `184Final.ipynb` in Jupyter Notebook or Google Colab
2. Ensure all required datasets are in the same directory
3. Run cells sequentially to perform the complete analysis

### Key Functions

#### Player Ranking Prediction
```python
# Train the model
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)

# Make predictions
y_pred = model.predict(xtest)
```

#### Trade Simulation
```python
# Simulate a trade between two teams
team1 = ["Player1", "Player2"]
team2 = ["Player3", "Player4"]
trade_simulator(team1, team2, trade_df)
```

#### Model Evaluation
```python
# Calculate accuracy metrics
mse = mean_squared_error(ytest, y_pred)
mae = mean_absolute_error(ytest, y_pred)
r2 = r2_score(ytest, y_pred)
```

## Dependencies

- **Python 3.7+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

## Installation

1. Clone or download the project files
2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```
3. Open Jupyter Notebook and navigate to the project directory

## Model Performance

The project evaluates model performance using:
- **Mean Squared Error (MSE)**: Measures prediction accuracy
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual rankings
- **R² Score**: Coefficient of determination for model fit
- **Tolerance Analysis**: Percentage of predictions within acceptable ranking ranges

## Future Enhancements

- **Real-time Data Integration**: Connect to live NBA statistics APIs
- **Advanced Analytics**: Implement more sophisticated machine learning models
- **Web Interface**: Create a user-friendly web application for analysis
- **Historical Analysis**: Expand to include more NBA seasons for trend analysis
- **Player Development Tracking**: Monitor player performance changes over time

## Contributing

This project is designed for educational and analytical purposes. Feel free to:
- Improve the machine learning models
- Add new statistical metrics
- Enhance the trade simulator
- Create additional visualization tools

## License

This project is for educational use. Please ensure compliance with any data usage terms from the original NBA statistics sources.

---

**Note**: This project analyzes publicly available NBA statistics for educational and analytical purposes. All player data and statistics are sourced from publicly available NBA information.
