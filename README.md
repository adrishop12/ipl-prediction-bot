# 🏏 IPL Match Prediction Bot

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0%2B-FF4B4B)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.0%2B-F7931E)]()

An advanced machine learning-powered IPL (Indian Premier League) cricket match prediction system that forecasts match outcomes and player performances using historical data and real-time analytics.

## 🎯 Features

- **Match Outcome Prediction**: Predicts winner between two teams using Random Forest Classifier
- **Player Performance Analysis**: Forecasts player performance using Linear Regression
- **Team Statistics**: Calculates win percentages and historical performance metrics
- **Real-time Updates**: Scrapes current team squads from iplt20.com
- **Interactive UI**: Built with Streamlit for easy interaction and visualization

## 🛠️ Technologies Used

- Python 3.8+
- Pandas & NumPy for data manipulation
- Scikit-learn for machine learning models
- Streamlit for web interface
- Beautiful Soup for web scraping
- Matplotlib & Seaborn for visualizations

## 📊 Data Sources

- `matches.csv`: Historical IPL match data
- `deliveries.csv`: Ball-by-ball match details
- `team_squads.json`: Current team compositions

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/adrishop12/ipl-prediction-bot.git
   cd ipl-prediction-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run ipl.py
   ```

## 🎮 How to Use

1. Select two teams from the dropdown
2. Choose the match venue
3. Select toss winner and decision
4. Get instant predictions for match outcomes
5. Explore player performance predictions

## 🧠 Model Details

- **Team Prediction Model**: Random Forest Classifier
  - Features: Team encodings, venue, toss details, historical win rates
  - Stored in: `models/team_model.pkl`

- **Player Performance Model**: Linear Regression
  - Features: Recent form, batting average, strike rate
  - Stored in: `models/player_model.pkl`

## 📝 Requirements

Check `requirements.txt` for detailed dependencies. Key requirements:
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- scikit-learn ≥ 1.2.0
- streamlit ≥ 1.22.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔍 Keywords
IPL prediction, cricket analytics, machine learning, sports prediction, IPL 2025, cricket match prediction, IPL match predictor, cricket statistics, sports analytics, Indian Premier League
