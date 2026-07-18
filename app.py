from flask import Flask, request, jsonify
from flask_cors import CORS
from services.aiService import DebateConductor
from services.mlJudge import DebateRegressionJudge

# Create the Flask app and allow the browser UI to communicate with it.
app = Flask(__name__)
CORS(app)

# Reuse the same debate controller and judge for every request.
conductor = DebateConductor()
ml_judge = DebateRegressionJudge()

# Train the regression judge when the server starts so evaluation is ready immediately.
try:
    ml_judge.train_model("historical_debates.csv")
except Exception as train_error:
    print(f"Warning: ML model could not be trained at startup: {train_error}")


@app.route("/api/debate/start", methods=["POST"])
def start_debate():
    """Start a new debate and generate the opening argument for Agent A."""
    data = request.json or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "A debate topic is required."}), 400

    try:
        # Reset any previous debate context before starting a new topic.
        conductor.reset_debate(topic)
        opening_message = conductor.generate_agent_a_response(topic)
        return jsonify({
            "status": "active",
            "topic": topic,
            "agent": "A",
            "message": opening_message,
        })
    except Exception as e:
        return jsonify({"error": f"Failed to start debate: {str(e)}"}), 500


@app.route("/api/debate/next-turn", methods=["POST"])
def next_turn():
    """Generate the next turn while keeping the debate history in memory."""
    data = request.json or {}
    topic = data.get("topic", conductor.current_topic).strip()
    last_speaker = data.get("last_speaker", "A")

    if not topic:
        return jsonify({"error": "Debate topic is missing."}), 400

    try:
        # Alternate turns so the debate continues naturally between the two agents.
        if last_speaker == "A":
            message = conductor.generate_agent_b_response(topic)
            agent = "B"
        else:
            message = conductor.generate_agent_a_response(topic)
            agent = "A"

        return jsonify({"agent": agent, "message": message})
    except Exception as e:
        return jsonify({"error": f"Turn generation failed: {str(e)}"}), 500


@app.route("/api/machine-learning/train", methods=["POST"])
def trigger_training():
    """Retrain the regression model from the CSV dataset if needed."""
    try:
        accuracy_metrics = ml_judge.train_model("historical_debates.csv")
        return jsonify({"status": "Training Completed", "metrics": accuracy_metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/machine-learning/evaluate", methods=["POST"])
def evaluate_debate():
    """Score both argument histories and return the winner."""
    data = request.json or {}
    advocate_text = data.get("advocate_text") or conductor.get_advocate_text()
    challenger_text = data.get("challenger_text") or conductor.get_challenger_text()

    if not advocate_text.strip() or not challenger_text.strip():
        return jsonify({"error": "Both advocate_text and challenger_text are required."}), 400

    try:
        # Score both argument histories using the trained regression model.
        advocate_score = ml_judge.predict_score(advocate_text)
        challenger_score = ml_judge.predict_score(challenger_text)

        if advocate_score >= challenger_score:
            winner = "Agent A (The Advocate)"
        else:
            winner = "Agent B (The Challenger)"

        return jsonify({
            "winner": winner,
            "advocate_score": advocate_score,
            "challenger_score": challenger_score,
            "metrics": ml_judge.get_metrics(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("AI Server running on http://127.0.0.1:5000")
    print("Ensure Ollama is running locally on port 11434 with model 'phi4-mini'!")
    app.run(debug=True, port=5000)
