# ⚽ FPL AI Optimizer

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, data-driven web application designed to help you dominate your Fantasy Premier League season. This tool uses player performance data and linear optimization to recommend the perfect 15-player squad, suggest weekly captains, and provide real-time gameweek tracking.


---

## ✨ Key Features

* **🤖 AI Squad Optimization:** Calculates the optimal 15-player squad within the £100m budget to maximize predicted points.
* **👁️ Visual Pitch Layout:** Displays your optimized team on a beautiful pitch layout with official team jerseys.
* **📈 Live Gameweek Tracker:** Follow your player points and live Premier League scores in real-time as the matches happen.
* **🧠 Dynamic Chip Strategy:** Get a data-driven recommendation on whether to use your Triple Captain, Bench Boost, or Free Hit each week.
* **🖥️ Interactive UI:** A clean and user-friendly interface built with Streamlit.

---

## 🛠️ Technology Stack

* **Backend:** Python
* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Optimization:** PuLP
* **APIs:** Official FPL API, football-data.org

---

## 🚀 Getting Started

Follow these steps to get the FPL AI Optimizer running on your local machine.

### 1. Prerequisites

* Python 3.9 or higher.
* A free API key from [football-data.org](https://www.football-data.org/login) for the live scoreboard.

### 2. Installation

Clone the repository to your local machine:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

Create a virtual environment and install the required packages:
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install streamlit pandas numpy requests pulp

3. Running the App
Launch the Streamlit application from your terminal:
streamlit run fpl_app.py

📄 License
This project is licensed under the MIT License.
