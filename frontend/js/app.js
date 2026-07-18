// --- PRODUCTION LOGIC (CONNECTS TO BACKEND) ---
let currentTopic = "";
let lastSpeaker = null;
let lastMessage = "";
let currentRound = 1;
const MAX_ROUNDS = 10;

const topicInput = document.getElementById('topicInput');
const startBtn = document.getElementById('startBtn');
const nextBtn = document.getElementById('nextTurnBtn');
const endBtn = document.getElementById('endDebateBtn');
const feedA = document.getElementById('feedA');
const feedB = document.getElementById('feedB');

const displayTopic = document.getElementById('displayTopic');
const networkStatus = document.getElementById('networkStatus');
const statusText = document.getElementById('statusText');
const dotA = document.getElementById('dotA');
const dotB = document.getElementById('dotB');
const globalDot = document.getElementById('globalDot');

const API_URL = "http://127.0.0.1:5000/api/debate";
const ML_URL = "http://127.0.0.1:5000/api/machine-learning";

function toggleTyping(agent, show) {
    const feed = agent === 'A' ? feedA : feedB;
    const existing = document.getElementById('typingIndicator');

    if (show) {
        if (!existing) {
            const typingHTML = `<div class="typing-wrapper" id="typingIndicator"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>`;
            feed.insertAdjacentHTML('beforeend', typingHTML);
            feed.scrollTop = feed.scrollHeight;
        }
    } else if (existing) {
        existing.remove();
    }
}

function setActive(agent) {
    if (agent === 'A') {
        dotA.classList.add('active');
        dotB.classList.remove('active');
        statusText.textContent = `Awaiting API Response for Agent A... (Round ${currentRound})`;
    } else {
        dotB.classList.add('active');
        dotA.classList.remove('active');
        statusText.textContent = `Awaiting API Response for Agent B... (Round ${currentRound})`;
    }
}

function appendMessage(agent, text, round) {
    toggleTyping(agent, false);

    const div = document.createElement('div');
    div.classList.add('msg', agent === 'A' ? 'msg-adv' : 'msg-chal');
    div.innerHTML = `
        <div class="round-tag">Round ${round} · ${agent === 'A' ? 'Advocate' : 'Challenger'}</div>
        <div class="msg-text">${text}</div>
    `;

    const feed = agent === 'A' ? feedA : feedB;
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;
}

function setProcessingState(isLoading) {
    startBtn.disabled = isLoading;
    nextBtn.disabled = isLoading || !lastSpeaker;
    endBtn.disabled = isLoading || !lastSpeaker;
    topicInput.disabled = isLoading;
    if (isLoading) {
        networkStatus.textContent = "COMPUTING";
        networkStatus.style.color = "#eab308";
        networkStatus.style.borderColor = "rgba(234,179,8,0.3)";
    } else {
        networkStatus.textContent = "IDLE";
        networkStatus.style.color = "#8b5cf6";
        networkStatus.style.borderColor = "rgba(139,92,246,0.3)";
    }
}

async function showMlVerdict() {
    statusText.textContent = "Debate concluded. ML Judge computing verdict...";
    setProcessingState(true);

    try {
        const response = await fetch(`${ML_URL}/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Evaluation failed");

        const sA = data.advocate_score.toFixed(1);
        const sB = data.challenger_score.toFixed(1);

        document.getElementById('scoreA').textContent = sA + ' / 10';
        document.getElementById('scoreB').textContent = sB + ' / 10';
        document.getElementById('barA').style.width = (parseFloat(sA) * 10) + '%';
        document.getElementById('barB').style.width = (parseFloat(sB) * 10) + '%';

        document.getElementById('verdictBox').innerHTML = `
            <strong>Winner: ${data.winner}</strong><br/><br/>
            The RandomForestRegressor analyzed all accumulated arguments from both agents.
            ${data.winner} achieved the higher predicted human persuasiveness score.
        `;

        const metrics = data.metrics || {};
        document.getElementById('judgeMetrics').innerHTML = `
            <div class="metric"><div class="label">Model Used</div><div class="val">RandomForestRegressor</div></div>
            <div class="metric"><div class="label">Features Extracted</div><div class="val">Word Count · Complexity</div></div>
            <div class="metric"><div class="label">Mean Squared Error</div><div class="val">${metrics.mse ?? 'N/A'}</div></div>
            <div class="metric"><div class="label">R² Accuracy</div><div class="val">${metrics.r2_score ?? 'N/A'}</div></div>
        `;

        document.getElementById('judgeOverlay').style.display = 'flex';
        globalDot.classList.remove('live');
        statusText.textContent = "ML Verdict delivered.";
    } catch (err) {
        console.error(err);
        alert("ML Judge evaluation failed: " + err.message);
    } finally {
        setProcessingState(false);
        nextBtn.disabled = true;
        endBtn.disabled = true;
    }
}

startBtn.addEventListener('click', async () => {
    const topic = topicInput.value.trim();
    if (!topic) return alert("Please enter a custom topic first.");

    currentTopic = topic;
    displayTopic.textContent = `"${topic}"`;
    feedA.innerHTML = '';
    feedB.innerHTML = '';
    currentRound = 1;
    lastSpeaker = null;
    lastMessage = "";

    setProcessingState(true);
    globalDot.classList.add('live');

    setActive('A');
    toggleTyping('A', true);

    try {
        const response = await fetch(`${API_URL}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic: currentTopic }),
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Start failed");

        lastSpeaker = data.agent || "A";
        lastMessage = data.message;

        appendMessage(lastSpeaker, lastMessage, currentRound);

        setProcessingState(false);
        nextBtn.disabled = false;
        endBtn.disabled = false;
        dotA.classList.remove('active');
        statusText.textContent = "API Idle. Pass turn to Agent B...";
    } catch (err) {
        console.error("Backend connection failed.", err);
        alert("Failed to connect to the AI Backend: " + err.message);
        setProcessingState(false);
        globalDot.classList.remove('live');
        toggleTyping('A', false);
    }
});

nextBtn.addEventListener('click', async () => {
    if (currentRound >= MAX_ROUNDS && lastSpeaker === 'B') {
        await showMlVerdict();
        return;
    }

    setProcessingState(true);

    const nextAgent = lastSpeaker === 'A' ? 'B' : 'A';
    setActive(nextAgent);
    toggleTyping(nextAgent, true);

    try {
        const response = await fetch(`${API_URL}/next-turn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topic: currentTopic,
                last_speaker: lastSpeaker,
                last_message: lastMessage,
            }),
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Turn failed");

        lastSpeaker = data.agent || nextAgent;
        lastMessage = data.message;

        appendMessage(lastSpeaker, lastMessage, currentRound);

        if (lastSpeaker === 'B') {
            currentRound++;
            if (currentRound > MAX_ROUNDS) {
                await showMlVerdict();
                return;
            }
        }

        setProcessingState(false);
        dotA.classList.remove('active');
        dotB.classList.remove('active');
        statusText.textContent = "API Idle. Waiting for User Execution...";
    } catch (err) {
        console.error(err);
        alert("Agent failed to respond: " + err.message);
        setProcessingState(false);
        toggleTyping(nextAgent, false);
    }
});

endBtn.addEventListener('click', showMlVerdict);