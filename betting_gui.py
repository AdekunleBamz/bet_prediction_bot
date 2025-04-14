import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                            QTabWidget, QLineEdit, QFormLayout, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from betting_bot import BettingBot
import asyncio
import logging
from datetime import datetime

class BotWorker(QThread):
    status_update = pyqtSignal(str)
    prediction_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.bot = None
        self.is_running = False

    def run(self):
        self.is_running = True
        asyncio.run(self.run_bot())

    async def run_bot(self):
        try:
            self.bot = BettingBot()
            self.status_update.emit("Initializing bot...")
            
            # Verify connections
            if not await self.bot.verify_connection():
                self.status_update.emit("Failed to verify connections. Check your internet and API keys.")
                return

            self.status_update.emit("Bot started successfully!")
            
            # Start processing matches
            while self.is_running:
                try:
                    now = datetime.now()
                    
                    # Make predictions at midnight
                    if now.hour == 0 and now.minute < 5:
                        self.status_update.emit("Making daily predictions...")
                        await self.bot.make_predictions()
                        await asyncio.sleep(300)
                    
                    # Check results every hour
                    if now.minute < 5:
                        self.status_update.emit("Checking match results...")
                        await self.bot.check_results()
                        await asyncio.sleep(300)
                    
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    self.status_update.emit(f"Error in bot loop: {str(e)}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            self.status_update.emit(f"Bot error: {str(e)}")

    def stop(self):
        self.is_running = False
        self.status_update.emit("Stopping bot...")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Betting Prediction Bot")
        self.setMinimumSize(800, 600)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Create tabs
        self.dashboard_tab = self.create_dashboard_tab()
        self.settings_tab = self.create_settings_tab()
        self.logs_tab = self.create_logs_tab()
        
        tabs.addTab(self.dashboard_tab, "Dashboard")
        tabs.addTab(self.settings_tab, "Settings")
        tabs.addTab(self.logs_tab, "Logs")
        
        # Initialize bot worker
        self.bot_worker = BotWorker()
        self.bot_worker.status_update.connect(self.update_status)
        self.bot_worker.prediction_update.connect(self.update_predictions)
        
        self.setup_styles()

    def create_dashboard_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Status section
        status_group = QWidget()
        status_layout = QHBoxLayout(status_group)
        
        self.status_label = QLabel("Status: Not Running")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.start_button = QPushButton("Start Bot")
        self.start_button.clicked.connect(self.start_bot)
        status_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Bot")
        self.stop_button.clicked.connect(self.stop_bot)
        self.stop_button.setEnabled(False)
        status_layout.addWidget(self.stop_button)
        
        layout.addWidget(status_group)
        
        # Predictions display
        predictions_label = QLabel("Latest Predictions")
        predictions_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(predictions_label)
        
        self.predictions_display = QTextEdit()
        self.predictions_display.setReadOnly(True)
        layout.addWidget(self.predictions_display)
        
        return widget

    def create_settings_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Telegram settings
        self.telegram_token = QLineEdit()
        self.telegram_token.setText(os.getenv('TELEGRAM_TOKEN', ''))
        layout.addRow("Telegram Token:", self.telegram_token)
        
        self.channel_username = QLineEdit()
        self.channel_username.setText(os.getenv('CHANNEL_USERNAME', ''))
        layout.addRow("Channel Username:", self.channel_username)
        
        # API settings
        self.api_key = QLineEdit()
        self.api_key.setText(os.getenv('RAPID_API_KEY', ''))
        layout.addRow("RapidAPI Key:", self.api_key)
        
        # Prediction settings
        self.max_predictions = QSpinBox()
        self.max_predictions.setRange(1, 50)
        self.max_predictions.setValue(30)
        layout.addRow("Max Predictions per Day:", self.max_predictions)
        
        # Save button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addRow("", save_button)
        
        return widget

    def create_logs_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)
        
        clear_button = QPushButton("Clear Logs")
        clear_button.clicked.connect(self.clear_logs)
        layout.addWidget(clear_button)
        
        return widget

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 8px;
            }
            QLabel {
                color: #212121;
            }
        """)

    def start_bot(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Starting...")
        self.bot_worker.start()

    def stop_bot(self):
        self.bot_worker.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped")

    def update_status(self, status):
        self.status_label.setText(f"Status: {status}")
        self.log_display.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {status}")

    def update_predictions(self, prediction):
        self.predictions_display.append(prediction)

    def save_settings(self):
        # Update environment variables
        os.environ['TELEGRAM_TOKEN'] = self.telegram_token.text()
        os.environ['CHANNEL_USERNAME'] = self.channel_username.text()
        os.environ['RAPID_API_KEY'] = self.api_key.text()
        
        # Update .env file
        with open('.env', 'w') as f:
            f.write(f'TELEGRAM_TOKEN={self.telegram_token.text()}\n')
            f.write(f'CHANNEL_USERNAME={self.channel_username.text()}\n')
            f.write(f'RAPID_API_KEY={self.api_key.text()}\n')
        
        self.update_status("Settings saved successfully!")

    def clear_logs(self):
        self.log_display.clear()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 