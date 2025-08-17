⚽ FPL AI Optimizer
An intelligent, data-driven web application designed to help you dominate your Fantasy Premier League season. This tool uses player performance data and linear optimization to recommend the perfect 15-player squad, suggest weekly captains, and provide real-time gameweek tracking.

(Replace this line with a screenshot of your running app!)

✨ Key Features
🤖 AI Squad Optimization: Calculates the optimal 15-player squad within the £100m budget to maximize predicted points.

視覺化球場佈局 (Visual Pitch Layout): Displays your optimized team on a beautiful pitch layout with official team jerseys.

📈 Live Gameweek Tracker: Follow your player points and live Premier League scores in real-time as the matches happen.

🧠 Dynamic Chip Strategy: Get a data-driven recommendation on whether to use your Triple Captain, Bench Boost, or Free Hit each week.

インタラクティブUI (Interactive UI): A clean and user-friendly interface built with Streamlit.

🛠️ Technology Stack
Backend: Python

Frontend: Streamlit

Data Processing: Pandas, NumPy

Optimization: PuLP

APIs: Official FPL API, football-data.org

🚀 Getting Started
Follow these steps to get the FPL AI Optimizer running on your local machine.

1. Prerequisites
Python 3.9 or higher.

A free API key from football-data.org for the live scoreboard.

2. Installation
Clone the repository to your local machine:

Bash

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
Create a virtual environment and install the required packages:

Bash

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install streamlit pandas numpy requests pulp
3. Configuration
The app is hardcoded with a working API key, but it's best practice to manage your own. The live score functionality depends on it.

4. Running the App
Launch the Streamlit application from your terminal:

Bash

streamlit run fpl_app.py
Your web browser will open with the app running locally!

license
This project is licensed under the MIT License. See the LICENSE file for details.
