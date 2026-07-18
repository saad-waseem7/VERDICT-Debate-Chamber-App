# Verdict

**Autonomous Debate Chamber**

Verdict is a lightweight AI project that pairs two conversational agents against each other on any topic, streams the exchange through a web interface, and scores the outcome with a machine learning judge. Enter a topic, watch the agents argue opposing sides in real time, and get a verdict backed by a trained regression model — not a coin flip.

## What It Does

Verdict combines three parts into one working demo:

- a Flask backend that handles debate requests and turn-taking
- a local Ollama-based agent system that generates each side's arguments
- a Scikit-Learn regression model that scores persuasiveness and declares a winner

## Key Features

- Start a debate on any custom topic
- Watch two agents — the Advocate and the Challenger — respond in alternating turns
- Full debate history is preserved, so later responses stay aware of what's already been argued
- Both sides are scored by a trained regression model, not a hardcoded rule
- Results are shown through a clean, desktop-optimized web interface

## Project Structure

- `app.py` — main Flask server and API routes
- `services/aiService.py` — debate orchestration, prompt setup, and Ollama communication
- `services/mlJudge.py` — feature extraction and regression-based scoring
- `frontend/` — the web interface (`index.html`, `css/styles.css`, `js/app.js`)
- `historical_debates.csv` — sample training data used by the judge

## Requirements

- Python 3.10 or newer
- Ollama installed and running locally
- A supported Ollama model, such as `phi4-mini`

## Setup

1. Create and activate a virtual environment
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Start Ollama and pull a model
   ```
   ollama serve
   ollama pull phi4-mini
   ```

4. Run the backend
   ```
   python app.py
   ```

5. Open the frontend in a browser
   ```
   frontend/index.html
   ```

## How It Works

1. You enter a debate topic.
2. The backend starts the discussion and generates the Advocate's opening argument.
3. Each following turn continues the exchange, with prior arguments kept in context.
4. Once the debate ends, the machine learning judge scores both sides and returns a final verdict.

## Repository Contents

Include in the repository:

- `app.py`
- `services/`
- `frontend/`
- `historical_debates.csv`
- `README.md`
- `requirements.txt`

## Notes

- The frontend is built for desktop screens; it is not optimized for mobile.
- To use a different Ollama model, update the `MODEL` value in `services/aiService.py`.
