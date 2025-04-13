# Betting Prediction Bot 🎯

An intelligent betting prediction bot that uses machine learning to analyze and predict sports match outcomes. The bot supports both football ⚽️ and basketball 🏀 predictions.

## Features

- Real-time match predictions using Machine Learning
- Support for multiple football leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League, Europa League)
- Basketball game predictions
- Automated daily predictions at midnight UTC
- Result tracking and model updates based on actual match outcomes
- Telegram channel integration for prediction delivery

## Requirements

- Python 3.7+
- Required packages (install via pip):
  ```
  python-telegram-bot
  requests
  pytz
  numpy
  scikit-learn
  joblib
  pandas
  aiohttp
  ```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/adekunlebamz/bet_prediction_bot.git
   cd bet_prediction_bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment variables:
   - TELEGRAM_TOKEN
   - RAPID_API_KEY
   - CHANNEL_USERNAME

4. Run the bot:
   ```bash
   python betting_bot.py
   ```

## Configuration

The bot uses the following APIs:
- API-Football for football matches
- API-Basketball for basketball games
- Telegram Bot API for message delivery

Make sure to set up your API keys and Telegram channel before running the bot.

## Machine Learning

The bot uses Random Forest Classifier for predictions with:
- Automated model training and updates
- Feature engineering based on match odds
- Confidence-based prediction filtering

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Created by @adekunlebamz 