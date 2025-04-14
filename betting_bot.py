import os
import logging
from datetime import datetime, timedelta
import pytz
import requests
import json
from telegram import Bot
import asyncio
import aiohttp
from typing import Dict, List, Any
from aiohttp import ClientTimeout
from telegram.error import TimedOut, NetworkError, RetryAfter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHANNEL_USERNAME = os.getenv('CHANNEL_USERNAME')
MAX_PREDICTIONS_PER_DAY = 30

# API-SPORTS Configuration
RAPID_API_KEY = os.getenv('RAPID_API_KEY')
FOOTBALL_API_HOST = "api-football-v1.p.rapidapi.com"
BASKETBALL_API_HOST = "api-basketball.p.rapidapi.com"

# Football Leagues
FOOTBALL_LEAGUES = [
    39,   # Premier League
    140,  # La Liga
    135,  # Serie A
    78,   # Bundesliga
    61,   # Ligue 1
    2,    # Champions League
    3     # Europa League
]

# ML Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
FOOTBALL_MODEL_PATH = os.path.join(MODEL_DIR, 'football_model.joblib')
BASKETBALL_MODEL_PATH = os.path.join(MODEL_DIR, 'basketball_model.joblib')

class MLPredictor:
    def __init__(self, sport: str):
        self.sport = sport
        self.model = None
        self.scaler = None
        self.load_or_create_model()

    def load_or_create_model(self):
        """Load existing model or create a new one"""
        model_path = FOOTBALL_MODEL_PATH if self.sport == 'football' else BASKETBALL_MODEL_PATH
        scaler_path = os.path.join(MODEL_DIR, f'{self.sport}_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded existing {self.sport} model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.create_new_model()
        else:
            self.create_new_model()

    def create_new_model(self):
        """Create a new RandomForest model"""
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        logger.info(f"Created new {self.sport} model")

    def prepare_features(self, match_data: Dict) -> np.ndarray:
        """Extract and prepare features from match data"""
        features = []
        
        try:
            # Extract relevant features
            home_odds = float(match_data.get('home_odds', 2.0))
            away_odds = float(match_data.get('away_odds', 2.0))
            
            features = [
                home_odds,
                away_odds,
                1/home_odds,  # implied probability
                1/away_odds,  # implied probability
                home_odds/away_odds  # odds ratio
            ]
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

        return np.array(features).reshape(1, -1)

    def predict(self, match_data: Dict) -> Dict:
        """Make prediction for a match"""
        try:
            features = self.prepare_features(match_data)
            if features is None:
                return None

            # Scale features
            if self.scaler is not None:
                scaled_features = self.scaler.transform(features)
            else:
                scaled_features = features
            
            # Make prediction
            pred_proba = self.model.predict_proba(scaled_features)[0]
            prediction = 'home' if pred_proba[1] > 0.5 else 'away'
            confidence = max(pred_proba)

            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'home': pred_proba[1],
                    'away': pred_proba[0]
                }
            }

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

    def update_model(self, match_data: Dict, actual_result: str):
        """Update model with new match result"""
        try:
            features = self.prepare_features(match_data)
            if features is None:
                return

            # Scale features
            if self.scaler is not None:
                scaled_features = self.scaler.transform(features)
            else:
                scaled_features = features
            
            # Convert result to numeric
            y = 1 if actual_result == 'home' else 0

            # Partial fit
            self.model.fit(scaled_features, [y])
            
            # Save updated model and scaler
            model_path = FOOTBALL_MODEL_PATH if self.sport == 'football' else BASKETBALL_MODEL_PATH
            scaler_path = os.path.join(MODEL_DIR, f'{self.sport}_scaler.joblib')
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Updated and saved {self.sport} model")

        except Exception as e:
            logger.error(f"Error updating model: {e}")

class BettingBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.predictions = {}
        self.headers = {
            "X-RapidAPI-Key": RAPID_API_KEY,
            "X-RapidAPI-Host": FOOTBALL_API_HOST
        }
        # Updated API endpoints
        self.football_base_url = "https://api-football-v1.p.rapidapi.com/v2"
        self.basketball_base_url = "https://api-basketball.p.rapidapi.com"
        self.timeout = ClientTimeout(total=30)
        
        # Initialize ML predictors
        self.football_predictor = MLPredictor('football')
        self.basketball_predictor = MLPredictor('basketball')

    async def send_message_with_retry(self, message: str, max_retries: int = 3):
        """Send message to the channel with retries"""
        for attempt in range(max_retries):
            try:
                await self.bot.send_message(
                    chat_id=CHANNEL_USERNAME,
                    text=message,
                    parse_mode='HTML'
                )
                logger.info("Message sent successfully")
                return True
            except TimedOut:
                logger.warning(f"Timeout on attempt {attempt + 1} of {max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                logger.error("Final timeout error sending message")
                return False
            except RetryAfter as e:
                logger.info(f"Rate limited. Waiting {e.retry_after} seconds")
                await asyncio.sleep(e.retry_after + 1)
                continue
            except NetworkError:
                logger.warning(f"Network error on attempt {attempt + 1} of {max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                logger.error("Network error sending message")
                return False
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                return False

    async def fetch_data_with_retry(self, url: str, headers: Dict, max_retries: int = 3) -> Dict:
        """Fetch data from APIs with retry mechanism"""
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', 60))
                            logger.info(f"Rate limited. Waiting {retry_after} seconds")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            logger.error(f"API request failed with status {response.status}")
                            return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1} of {max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                logger.error(f"Timeout error fetching data from {url}")
                return None
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                return None
        return None

    async def verify_connection(self) -> bool:
        """Verify bot can connect to Telegram and API services"""
        try:
            # Test Telegram connection
            me = await self.bot.get_me()
            logger.info(f"Connected to Telegram as {me.username}")
            
            # Test Football API connection
            football_headers = self.headers.copy()
            football_headers["X-RapidAPI-Host"] = FOOTBALL_API_HOST
            today = datetime.now().strftime('%Y-%m-%d')
            football_test = await self.fetch_data_with_retry(
                f"{self.football_base_url}/fixtures/date/{today}",
                football_headers
            )
            if not football_test:
                logger.error("Could not connect to Football API")
                return False
            logger.info("Successfully connected to Football API")
            
            # Test Basketball API connection
            basketball_headers = self.headers.copy()
            basketball_headers["X-RapidAPI-Host"] = BASKETBALL_API_HOST
            basketball_test = await self.fetch_data_with_retry(
                f"{self.basketball_base_url}/games?date={today}",
                basketball_headers
            )
            if not basketball_test:
                logger.error("Could not connect to Basketball API")
                return False
            logger.info("Successfully connected to Basketball API")
            
            logger.info("All connections verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return False

    async def get_football_odds(self, league_id: int) -> List[Dict]:
        """Fetch football odds for a specific league"""
        headers = self.headers.copy()
        headers["X-RapidAPI-Host"] = FOOTBALL_API_HOST
        url = f"{self.football_base_url}/odds/league/{league_id}/bookmaker/5"
        
        data = await self.fetch_data_with_retry(url, headers)
        if data and 'api' in data and 'odds' in data['api']:
            return data['api']['odds']
        return []

    async def get_basketball_games(self) -> List[Dict]:
        """Fetch basketball games"""
        headers = self.headers.copy()
        headers["X-RapidAPI-Host"] = BASKETBALL_API_HOST
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"{self.basketball_base_url}/games?date={today}"
        
        data = await self.fetch_data_with_retry(url, headers)
        if data and 'response' in data:
            return data['response']
        return []

    async def analyze_football_odds(self, odds: Dict) -> Dict:
        """Analyze football odds and make prediction using ML model"""
        try:
            if not odds.get('bookmakers'):
                return None

            bookmaker = odds['bookmakers'][0]
            if not bookmaker.get('bets'):
                return None

            match_odds = bookmaker['bets'][0]['values']
            home_odds = float(match_odds[0]['odd'])
            away_odds = float(match_odds[1]['odd'])

            # Prepare match data for ML prediction
            match_data = {
                'home_odds': home_odds,
                'away_odds': away_odds
            }

            # Get ML prediction
            ml_prediction = self.football_predictor.predict(match_data)
            if not ml_prediction:
                return None

            prediction = {
                'sport': 'football',
                'match_id': odds['fixture']['id'],
                'home_team': odds['fixture']['homeTeam'],
                'away_team': odds['fixture']['awayTeam'],
                'league': odds['league']['name'],
                'time': datetime.fromtimestamp(odds['fixture']['timestamp']),
                'prediction': ml_prediction['prediction'],
                'confidence': ml_prediction['confidence'],
                'ml_probabilities': ml_prediction['probabilities']
            }
            
            return prediction
        except Exception as e:
            logger.error(f"Error analyzing football odds: {e}")
            return None

    async def analyze_basketball_game(self, game: Dict) -> Dict:
        """Analyze basketball game using ML model"""
        try:
            # For demonstration, using placeholder odds (you should fetch real odds)
            match_data = {
                'home_odds': 1.95,  # placeholder
                'away_odds': 1.95   # placeholder
            }

            # Get ML prediction
            ml_prediction = self.basketball_predictor.predict(match_data)
            if not ml_prediction:
                return None

            prediction = {
                'sport': 'basketball',
                'match_id': game['id'],
                'home_team': game['teams']['home']['name'],
                'away_team': game['teams']['away']['name'],
                'league': game['league']['name'],
                'time': datetime.fromisoformat(game['date'].replace('Z', '+00:00')),
                'prediction': ml_prediction['prediction'],
                'confidence': ml_prediction['confidence'],
                'ml_probabilities': ml_prediction['probabilities']
            }
            
            return prediction
        except Exception as e:
            logger.error(f"Error analyzing basketball game: {e}")
            return None

    async def make_predictions(self):
        """Make daily predictions for matches"""
        try:
            all_predictions = []
            
            # Get football predictions
            for league_id in FOOTBALL_LEAGUES:
                odds_list = await self.get_football_odds(league_id)
                for odds in odds_list:
                    prediction = await self.analyze_football_odds(odds)
                    if prediction and prediction['confidence'] > 0.55:
                        all_predictions.append(prediction)
            
            # Get basketball predictions
            games = await self.get_basketball_games()
            for game in games:
                prediction = await self.analyze_basketball_game(game)
                if prediction and prediction['confidence'] > 0.55:
                    all_predictions.append(prediction)

            # Sort predictions by confidence and take top MAX_PREDICTIONS_PER_DAY
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            selected_predictions = all_predictions[:MAX_PREDICTIONS_PER_DAY]

            if selected_predictions:
                message = "🎯 Today's Top Betting Predictions 🎯\n\n"
                message += "⚠️ Betting Advice: Never bet more than you can afford to lose.\n"
                message += "Recommended: Bet no more than 1-2% of your bankroll per game.\n\n"
                
                for idx, pred in enumerate(selected_predictions, 1):
                    emoji = "⚽️" if pred['sport'] == 'football' else "🏀"
                    message += f"{idx}. {emoji} {pred['league']}\n"
                    message += f"🏟 {pred['home_team']} vs {pred['away_team']}\n"
                    message += f"🕒 {pred['time'].strftime('%H:%M')} UTC\n"
                    message += f"📊 Prediction: {pred['prediction'].upper()} WIN\n"
                    message += f"💪 Confidence: {pred['confidence']*100:.1f}%\n\n"
                    
                    # Store prediction for later result checking
                    self.predictions[str(pred['match_id'])] = pred
                
                await self.send_message_with_retry(message)
            else:
                logger.info("No matches found for today")

        except Exception as e:
            logger.error(f"Error making predictions: {e}")

    async def check_results(self):
        """Check results and update ML models"""
        completed_matches = []
        
        for match_id, prediction in self.predictions.items():
            try:
                if prediction['sport'] == 'football':
                    headers = self.headers.copy()
                    headers["X-RapidAPI-Host"] = FOOTBALL_API_HOST
                    url = f"{self.football_base_url}/fixtures/id/{match_id}"
                else:  # basketball
                    headers = self.headers.copy()
                    headers["X-RapidAPI-Host"] = BASKETBALL_API_HOST
                    url = f"{self.basketball_base_url}/games?id={match_id}"
                
                match_data = await self.fetch_data_with_retry(url, headers)
                
                if match_data:
                    if prediction['sport'] == 'football':
                        fixture = match_data['api']['fixtures'][0]
                        if fixture['status'] == 'Match Finished':
                            home_score = fixture['goalsHomeTeam']
                            away_score = fixture['goalsAwayTeam']
                            completed_matches.append(match_id)
                            
                            result = 'home' if home_score > away_score else 'away'
                            prediction_correct = result == prediction['prediction']
                            
                            # Update ML model
                            self.football_predictor.update_model(
                                {'home_odds': prediction.get('home_odds', 2.0),
                                 'away_odds': prediction.get('away_odds', 2.0)},
                                result
                            )
                            
                            result_message = (
                                f"⚽️ Match Result:\n"
                                f"{prediction['home_team']} {home_score} - {away_score} {prediction['away_team']}\n"
                                f"Prediction: {prediction['prediction'].upper()} WIN\n"
                                f"Confidence: {prediction['confidence']*100:.1f}%\n"
                                f"Status: {'✅ CORRECT' if prediction_correct else '❌ INCORRECT'}\n"
                            )
                            await self.send_message_with_retry(result_message)
                    
                    else:  # basketball
                        game = match_data['response'][0]
                        if game['status']['long'] == 'Finished':
                            home_score = game['scores']['home']['total']
                            away_score = game['scores']['away']['total']
                            completed_matches.append(match_id)
                            
                            result = 'home' if home_score > away_score else 'away'
                            prediction_correct = result == prediction['prediction']
                            
                            # Update ML model
                            self.basketball_predictor.update_model(
                                {'home_odds': prediction.get('home_odds', 2.0),
                                 'away_odds': prediction.get('away_odds', 2.0)},
                                result
                            )
                            
                            result_message = (
                                f"🏀 Match Result:\n"
                                f"{prediction['home_team']} {home_score} - {away_score} {prediction['away_team']}\n"
                                f"Prediction: {prediction['prediction'].upper()} WIN\n"
                                f"Confidence: {prediction['confidence']*100:.1f}%\n"
                                f"Status: {'✅ CORRECT' if prediction_correct else '❌ INCORRECT'}\n"
                            )
                            await self.send_message_with_retry(result_message)
            
            except Exception as e:
                logger.error(f"Error checking match result: {e}")
        
        # Remove completed matches from predictions
        for match_id in completed_matches:
            self.predictions.pop(match_id, None)

    async def process_matches(self):
        """Process matches and send predictions"""
        while True:
            try:
                now = datetime.now(pytz.UTC)
                
                # Make predictions at midnight UTC
                if now.hour == 0 and now.minute < 5:
                    logger.info("Making daily predictions")
                    await self.make_predictions()
                    await asyncio.sleep(300)  # Wait 5 minutes after making predictions
                
                # Check results every hour
                if now.minute < 5:
                    logger.info("Checking match results")
                    await self.check_results()
                    await asyncio.sleep(300)  # Wait 5 minutes after checking results
                
                await asyncio.sleep(60)  # Check every minute otherwise
                
            except Exception as e:
                logger.error(f"Error in process_matches: {e}")
                await asyncio.sleep(60)

    async def run(self):
        """Run the bot"""
        try:
            # Verify connections first
            if not await self.verify_connection():
                logger.error("Failed to verify connections. Please check your internet and API keys.")
                return

            # Send startup message
            startup_success = await self.send_message_with_retry(
                "🤖 Betting Prediction Bot Started!\n\n"
                "I will send daily predictions at midnight UTC.\n"
                "Stay tuned for the best betting predictions! 🎯"
            )

            if not startup_success:
                logger.error("Failed to send startup message. Please check your Telegram token and channel username.")
                return

            logger.info("Bot started successfully!")
            
            # Start processing matches
            await self.process_matches()
            
        except Exception as e:
            logger.error(f"Error running bot: {e}")

async def main():
    while True:
        try:
            bot = BettingBot()
            await bot.run()
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            await asyncio.sleep(60)  # Wait a minute before restarting

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
