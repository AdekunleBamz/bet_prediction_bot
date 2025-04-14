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
MAX_PREDICTIONS_PER_DAY = 15

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
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
FOOTBALL_MODEL_PATH = os.path.join(MODEL_DIR, 'football_model.joblib')
BASKETBALL_MODEL_PATH = os.path.join(MODEL_DIR, 'basketball_model.joblib')

# Add these constants at the top of the file after other constants
API_RATE_LIMIT_DELAY = 3  # Seconds between API calls
MAX_RETRIES = 5
RETRY_DELAY = 60  # Seconds to wait after hitting rate limit

# Updated betting markets with 1xBet options
FOOTBALL_MARKETS = {
    'match_result': '1X2',  # Home(1), Draw(X), Away(2)
    'over_under_2_5': 'O/U 2.5',
    'btts': 'BTTS',
    'draw_no_bet': 'DNB',
    'handicap': 'HDP',
    'correct_score': 'CS',
    'double_chance': 'DC',  # 1X, X2, 12
    'first_half_result': '1H 1X2'
}

BASKETBALL_MARKETS = {
    'match_winner': 'ML',  # Money Line (Home/Away)
    'total_points': 'O/U',  # Over/Under total points
    'draw_no_bet': 'DNB',
    'handicap': 'HDP',
    'correct_score': 'CS',
    'first_quarter_winner': '1Q ML',
    'first_half_winner': '1H ML'
}

class MLPredictor:
    def __init__(self, sport: str):
        self.sport = sport
        self.models = {}
        self.scalers = {}
        
        # Define prediction types for each sport
        self.football_predictions = [
            'match_result',
            'first_half_result',
            'btts',
            'over_under_2_5',
            'over_under_1_5',
            'over_under_3_5',
            'first_half_over_under_1_5',
            'double_chance',
            'draw_no_bet',
            'both_teams_score_first_half',
            'first_team_score',
            'home_team_over_1_5',
            'away_team_over_1_5',
            'home_win_both_halves',
            'away_win_both_halves',
            'win_to_nil_home',
            'win_to_nil_away',
            'home_score_both_halves',
            'away_score_both_halves',
            'exact_goals_1_2',
            'exact_goals_2_3',
            'exact_goals_3_4'
        ]
        
        self.basketball_predictions = [
            'match_winner',
            'total_points_over_under',
            'first_quarter_winner',
            'first_half_winner',
            'point_spread',
            'first_quarter_total',
            'first_half_total',
            'home_team_total_over',
            'away_team_total_over',
            'winning_margin_1_10',
            'winning_margin_11_plus',
            'race_to_20_points',
            'highest_scoring_half',
            'team_highest_scoring_quarter',
            'will_be_overtime',
            'first_to_score',
            'last_to_score'
        ]
        
        self.load_or_create_models()

    def load_or_create_models(self):
        """Load existing models or create new ones"""
        prediction_types = (
            self.football_predictions if self.sport == 'football' 
            else self.basketball_predictions
        )
        
        for pred_type in prediction_types:
            model_path = os.path.join(MODEL_DIR, f'{self.sport}_{pred_type}_model.joblib')
            scaler_path = os.path.join(MODEL_DIR, f'{self.sport}_{pred_type}_scaler.joblib')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[pred_type] = joblib.load(model_path)
                    self.scalers[pred_type] = joblib.load(scaler_path)
                    logger.info(f"Loaded existing {self.sport} {pred_type} model")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    self.create_new_model(pred_type)
            else:
                self.create_new_model(pred_type)

    def create_new_model(self, pred_type: str):
        """Create a new RandomForest model for specific prediction type"""
        self.models[pred_type] = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.scalers[pred_type] = StandardScaler()
        
        # Initialize scaler with some default data
        if self.sport == 'football':
            default_data = np.array([
                [2.0, 2.0, 1.5, 1.2, 5.5, 4.5, 1.1],  # Default features for football
                [1.8, 2.2, 1.3, 1.4, 5.0, 5.0, 1.2],
                [2.1, 1.9, 1.6, 1.1, 6.0, 4.0, 1.0]
            ])
        else:  # basketball
            default_data = np.array([
                [2.0, 2.0, 105.5, 102.3, 26.5, 25.8],  # Default features for basketball
                [1.8, 2.2, 108.0, 100.0, 27.0, 25.0],
                [2.1, 1.9, 103.0, 104.5, 26.0, 26.5]
            ])
        
        self.scalers[pred_type].fit(default_data)
        # Train model with some default outcomes (balanced classes)
        default_outcomes = np.array([0, 1, 0])  # Example outcomes
        self.models[pred_type].fit(default_data, default_outcomes)
        
        logger.info(f"Created new {self.sport} {pred_type} model")

    def prepare_features(self, match_data: Dict, pred_type: str) -> np.ndarray:
        """Extract and prepare features from match data for specific prediction type"""
        try:
            # Base features for all predictions
            features = [
                float(match_data.get('home_odds', 2.0)),
                float(match_data.get('away_odds', 2.0)),
                float(match_data.get('home_form', 0.5)),  # Recent form (0-1)
                float(match_data.get('away_form', 0.5)),  # Recent form (0-1)
                float(match_data.get('h2h_advantage', 0.5)),  # Head-to-head advantage (0-1)
                float(match_data.get('league_position_diff', 0.0))  # League position difference
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing features for {pred_type}: {e}")
            return None

    def predict(self, match_data: Dict) -> Dict:
        """Make predictions for all relevant markets"""
        predictions = {}
        prediction_types = (
            self.football_predictions if self.sport == 'football' 
            else self.basketball_predictions
        )
        
        try:
            for pred_type in prediction_types:
                features = self.prepare_features(match_data, pred_type)
                if features is None:
                    continue

                # Scale features
                if self.scalers.get(pred_type) is not None:
                    scaled_features = self.scalers[pred_type].transform(features)
                else:
                    scaled_features = features
                
                # Make prediction
                model = self.models.get(pred_type)
                if model is None:
                    continue
                    
                pred_proba = model.predict_proba(scaled_features)[0]
                
                # Format prediction based on type
                if pred_type == 'match_result':
                    prediction = 'home' if pred_proba[1] > 0.5 else 'away'
                elif pred_type == 'btts':
                    prediction = 'Yes' if pred_proba[1] > 0.5 else 'No'
                elif pred_type.startswith('over_under'):
                    prediction = 'Over' if pred_proba[1] > 0.5 else 'Under'
                else:
                    prediction = 'Yes' if pred_proba[1] > 0.5 else 'No'

                confidence = max(pred_proba)
                
                if confidence > 0.55:  # Only include confident predictions
                    predictions[pred_type] = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'probabilities': {
                            'yes/home/over': pred_proba[1],
                            'no/away/under': pred_proba[0]
                        }
                    }

            return predictions if predictions else None

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

    def update_model(self, match_data: Dict, results: Dict):
        """Update models with new match results"""
        try:
            for pred_type, result in results.items():
                features = self.prepare_features(match_data, pred_type)
                if features is None:
                    continue

                # Scale features
                if self.scalers.get(pred_type) is not None:
                    scaled_features = self.scalers[pred_type].transform(features)
                else:
                    scaled_features = features
                
                # Convert result to numeric
                y = 1 if result in ['home', 'Yes', 'Over'] else 0

                # Update model
                model = self.models.get(pred_type)
                if model is None:
                    continue
                    
                model.fit(scaled_features, [y])
                
                # Save updated model and scaler
                model_path = os.path.join(MODEL_DIR, f'{self.sport}_{pred_type}_model.joblib')
                scaler_path = os.path.join(MODEL_DIR, f'{self.sport}_{pred_type}_scaler.joblib')
                os.makedirs(MODEL_DIR, exist_ok=True)
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[pred_type], scaler_path)
                
            logger.info(f"Updated and saved {self.sport} models")

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

    async def fetch_data_with_retry(self, url: str, headers: Dict, max_retries: int = MAX_RETRIES) -> Dict:
        """Fetch data from APIs with improved retry mechanism and rate limiting"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Add delay between API calls to respect rate limits
                if retry_count > 0:
                    await asyncio.sleep(API_RATE_LIMIT_DELAY)
                
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', RETRY_DELAY))
                            logger.info(f"Rate limited. Waiting {retry_after} seconds")
                            await asyncio.sleep(retry_after)
                            retry_count += 1
                            continue
                        else:
                            logger.error(f"API request failed with status {response.status}")
                            return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {retry_count + 1} of {max_retries}")
                if retry_count < max_retries - 1:
                    await asyncio.sleep(API_RATE_LIMIT_DELAY * (retry_count + 1))
                    retry_count += 1
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
        """Fetch football odds for a specific league with rate limiting"""
        headers = self.headers.copy()
        headers["X-RapidAPI-Host"] = FOOTBALL_API_HOST
        url = f"{self.football_base_url}/odds/league/{league_id}/bookmaker/5"
        
        # Add delay between league requests
        await asyncio.sleep(API_RATE_LIMIT_DELAY)
        
        data = await self.fetch_data_with_retry(url, headers)
        if data and 'api' in data and 'odds' in data['api']:
            return data['api']['odds']
        return []

    async def get_basketball_games(self) -> List[Dict]:
        """Fetch basketball games with rate limiting"""
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

            # Get historical stats (placeholder values for now)
            home_goals_avg = 1.5  # This should come from historical data
            away_goals_avg = 1.2
            home_corners_avg = 5.5
            away_corners_avg = 4.5
            first_half_goals_avg = 1.1

            # Prepare base match data
            match_data = {
                'home_odds': home_odds,
                'away_odds': away_odds,
                'home_goals_scored_avg': home_goals_avg,
                'away_goals_scored_avg': away_goals_avg,
                'home_corners_avg': home_corners_avg,
                'away_corners_avg': away_corners_avg,
                'first_half_goals_avg': first_half_goals_avg
            }

            # Get ML predictions for all markets
            predictions = self.football_predictor.predict(match_data)
            if not predictions:
                return None

            prediction = {
                'sport': 'football',
                'match_id': odds['fixture']['id'],
                'home_team': odds['fixture']['homeTeam'],
                'away_team': odds['fixture']['awayTeam'],
                'league': odds['league']['name'],
                'time': datetime.fromtimestamp(odds['fixture']['timestamp']),
                'predictions': predictions,
                'home_odds': home_odds,
                'away_odds': away_odds
            }
            
            return prediction
        except Exception as e:
            logger.error(f"Error analyzing football odds: {e}")
            return None

    async def analyze_basketball_game(self, game: Dict) -> Dict:
        """Analyze basketball game using ML model"""
        try:
            # Get odds (placeholder values for now)
            home_odds = 1.95
            away_odds = 1.95

            # Get historical stats (placeholder values for now)
            home_points_avg = 105.5
            away_points_avg = 102.3
            home_first_quarter_avg = 26.5
            away_first_quarter_avg = 25.8

            # Prepare match data
            match_data = {
                'home_odds': home_odds,
                'away_odds': away_odds,
                'home_points_avg': home_points_avg,
                'away_points_avg': away_points_avg,
                'home_first_quarter_avg': home_first_quarter_avg,
                'away_first_quarter_avg': away_first_quarter_avg
            }

            # Get ML predictions for all markets
            predictions = self.basketball_predictor.predict(match_data)
            if not predictions:
                return None

            prediction = {
                'sport': 'basketball',
                'match_id': game['id'],
                'home_team': game['teams']['home']['name'],
                'away_team': game['teams']['away']['name'],
                'league': game['league']['name'],
                'time': datetime.fromisoformat(game['date'].replace('Z', '+00:00')),
                'predictions': predictions,
                'home_odds': home_odds,
                'away_odds': away_odds
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
                    if prediction:
                        all_predictions.append(prediction)
            
            # Get basketball predictions
            games = await self.get_basketball_games()
            for game in games:
                prediction = await self.analyze_basketball_game(game)
                if prediction:
                    all_predictions.append(prediction)

            if not all_predictions:
                logger.info("No matches found for today. Waiting for next update at midnight UTC.")
                next_midnight = (datetime.now(pytz.UTC) + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                wait_seconds = (next_midnight - datetime.now(pytz.UTC)).total_seconds()
                await asyncio.sleep(wait_seconds)
                return

            # Sort predictions by confidence
            all_predictions.sort(
                key=lambda x: max(pred['confidence'] for pred in x['predictions'].values()),
                reverse=True
            )
            selected_predictions = all_predictions[:MAX_PREDICTIONS_PER_DAY]

            current_date = datetime.now().strftime('%Y-%m-%d')
            messages = []
            current_message = (
                f"🎯 Premium Tips for {current_date}\n\n"
                f"💫 Top {MAX_PREDICTIONS_PER_DAY} Predictions Today\n\n"
            )
            
            for idx, pred in enumerate(selected_predictions, 1):
                # Sport-specific emoji
                sport_emoji = "⚽️" if pred['sport'] == 'football' else "🏀"
                
                # Sort predictions by confidence
                sorted_predictions = sorted(
                    pred['predictions'].items(),
                    key=lambda x: x[1]['confidence'],
                    reverse=True
                )
                
                # Format predictions with percentages
                predictions_list = []
                for market, market_pred in sorted_predictions:
                    if market_pred['confidence'] * 100 >= 55:  # Only show predictions with >55% confidence
                        confidence_pct = market_pred['confidence'] * 100
                        market_name = (FOOTBALL_MARKETS.get(market, '') 
                                     if pred['sport'] == 'football' 
                                     else BASKETBALL_MARKETS.get(market, ''))
                        pred_str = f"• {market_name}: {market_pred['prediction']}"
                        if 'odds' in market_pred:
                            pred_str += f" @{market_pred['odds']}"
                        pred_str += f" ({confidence_pct:.1f}%)"
                        predictions_list.append(pred_str)
                
                if predictions_list:
                    match_message = (
                        f"{idx}. {sport_emoji} {pred['league']}\n"
                        f"🏟 {pred['home_team']} vs {pred['away_team']}\n"
                        f"⏰ {pred['time'].strftime('%H:%M')} UTC\n\n"
                        f"{chr(10).join(predictions_list)}\n\n"
                        f"〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️\n\n"
                    )
                    
                    if len(current_message + match_message) > 4000:
                        messages.append(current_message)
                        current_message = match_message
                    else:
                        current_message += match_message
            
            # Add footer with minimal information
            footer = (
                "\n📱 @bamzz_cryptoalpha\n"
                "⚠️ Stake 1-2% per tip\n"
                "🔄 Next update: 00:00 UTC"
            )
            
            if len(current_message + footer) <= 4000:
                current_message += footer
                messages.append(current_message)
            else:
                messages.append(current_message)
                messages.append(footer)
            
            # Send all messages
            for message in messages:
                success = await self.send_message_with_retry(message)
                if not success:
                    logger.error("Failed to send predictions message")
                    break
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error making predictions: {e}")

    async def check_results(self):
        """Check results and update ML models with rate limiting"""
        completed_matches = []
        
        for match_id, prediction in self.predictions.items():
            try:
                # Add delay between result checks
                await asyncio.sleep(API_RATE_LIMIT_DELAY)
                
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
        try:
            # Force immediate prediction on startup
            logger.info("Making initial predictions...")
            await self.make_predictions()
            
            while True:
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
