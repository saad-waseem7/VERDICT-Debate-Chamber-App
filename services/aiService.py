import requests


class DebateConductor:
    """Orchestrates a two-agent debate through a local Ollama chat endpoint."""

    OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
    MODEL = "phi4-mini"

    # These prompts keep each agent focused on defending or attacking the topic.
    AGENT_A_SYSTEM_PROMPT = (
        "You are Agent A: The Advocate. You fiercely defend the assigned topic. "
        "Style: confident, logical, and concise. "
        "CRITICAL: Directly refute the opponent's last argument using completely different words. "
        "Never repeat their exact sentences, phrasing, or vocabulary. "
        "Limit your response strictly to 1 or 2 brief paragraphs (maximum 100 words). "
        "Do not write your name, role, or introduce yourself. Begin arguing immediately."
    )

    AGENT_B_SYSTEM_PROMPT = (
        "You are Agent B: The Challenger. You fiercely oppose the assigned topic. "
        "Style: critical, sharp, and concise. "
        "CRITICAL: Directly refute the opponent's last argument using completely different words. "
        "Never repeat their exact sentences, phrasing, or vocabulary. "
        "Limit your response strictly to 1 or 2 brief paragraphs (maximum 100 words). "
        "Do not write your name, role, or introduce yourself. Begin arguing immediately."
    )

    def __init__(self):
        self.ollama_url = self.OLLAMA_CHAT_URL
        self.model = self.MODEL
        self.debate_history = []
        self.current_topic = ""

    def reset_debate(self, topic):
        """Clear prior turns and set a fresh topic for the next debate."""
        # Reset the stored history so the next debate starts cleanly.
        self.debate_history = []
        self.current_topic = topic

    def _normalize_messages_for_agent(self, running_agent, topic):
        """Build a clean chat history that makes each agent answer in a consistent way."""
        system_content = (
            self.AGENT_A_SYSTEM_PROMPT if running_agent == "A" else self.AGENT_B_SYSTEM_PROMPT
        )

        messages = [{"role": "system", "content": system_content}]

        # The first turn needs a simple opening prompt instead of prior debate history.
        if not self.debate_history:
            messages.append({
                "role": "user",
                "content": f"The debate topic is: '{topic}'. Provide your opening statement now."
            })
            return messages

        # Convert the saved debate turns into a structured chat format.
        for turn in self.debate_history:
            if turn["agent"] == running_agent:
                role = "assistant"
            else:
                role = "user"
            messages.append({"role": role, "content": turn["message"]})

        # Add one final instruction so the model stays focused on the next counterpoint.
        messages.append({
            "role": "user",
            "content": "Deliver your next counter-argument. Do not echo or mirror the phrasing of the previous point."
        })

        return messages

    def _call_ollama(self, messages):
        """Send a non-streaming request to the local Ollama instance."""
        # These options keep the responses short, focused, and suitable for the UI.
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.75,
                "num_predict": 140,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.4,
            },
        }

        response = requests.post(self.ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("Ollama returned an empty response.")

        return self._clean_truncated_sentence(content)

    def _clean_truncated_sentence(self, text):
        """Trim unfinished sentences if the model stops early."""
        if not text:
            return text
        last_punctuation = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punctuation != -1:
            return text[:last_punctuation + 1]
        return text

    def _record_turn(self, agent, message):
        # Store each reply so later turns can build on the ongoing discussion.
        self.debate_history.append({"agent": agent, "message": message})

    def generate_agent_a_response(self, topic):
        """Generate a response for Agent A, the advocate."""
        messages = self._normalize_messages_for_agent("A", topic)
        reply = self._call_ollama(messages)
        self._record_turn("A", reply)
        return reply

    def generate_agent_b_response(self, topic):
        """Generate a response for Agent B, the challenger."""
        messages = self._normalize_messages_for_agent("B", topic)
        reply = self._call_ollama(messages)
        self._record_turn("B", reply)
        return reply

    def get_advocate_text(self):
        """Collect all Agent A arguments for scoring."""
        return " ".join(
            turn["message"] for turn in self.debate_history if turn["agent"] == "A"
        )

    def get_challenger_text(self):
        """Collect all Agent B arguments for scoring."""
        return " ".join(
            turn["message"] for turn in self.debate_history if turn["agent"] == "B"
        )