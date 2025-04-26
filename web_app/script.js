'use strict';

// --- DOM Elements ---
const interviewIdInput = document.getElementById('interviewId');
const startInterviewBtn = document.getElementById('startInterviewBtn');
const videoModeSelect = document.getElementById('videoMode');
const stopInterviewBtn = document.getElementById('stopInterviewBtn');
const localVideo = document.getElementById('localVideo');
const mediaStatus = document.getElementById('mediaStatus');
const agentTextOutput = document.getElementById('agentTextOutput');
const chatOutput = document.getElementById('chatOutput');
const chatInput = document.getElementById('chatInput');
const sendChatBtn = document.getElementById('sendChatBtn');
const hrIdInput = document.getElementById('hrId');
const getSummaryBtn = document.getElementById('getSummaryBtn');
const interviewSummary = document.getElementById('interviewSummary');

// --- State Variables ---
let localStream = null;
let webSocket = null; // Placeholder for WebSocket connection
let audioContext = null;
let audioWorkletNode = null;
let mediaRecorder = null; // For sending audio chunks
const audioChunkQueue = [];
let isInterviewActive = false;

// --- Constants ---
const BACKEND_WS_URL = 'ws://localhost:8080/ws'; // Example WebSocket URL - NEEDS A BACKEND
const AUDIO_SAMPLE_RATE = 16000; // Matches Python script SEND_SAMPLE_RATE
const AUDIO_CHUNK_DURATION_MS = 100; // Send audio chunks every 100ms

// --- Functions ---

function logToChat(message, sender = 'System') {
    const messageElement = document.createElement('p');
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`; // Use innerHTML to render potential formatting
    chatOutput.appendChild(messageElement);
    chatOutput.scrollTop = chatOutput.scrollHeight; // Scroll to bottom
}

function logAgentResponse(text) {
    // In a real implementation, we might receive structured data
    // For now, just append the text
    agentTextOutput.innerHTML += `<p>${text}</p>`;
    agentTextOutput.scrollTop = agentTextOutput.scrollHeight;
}

function setUIState(active) {
    isInterviewActive = active;
    startInterviewBtn.disabled = active;
    stopInterviewBtn.disabled = !active;
    interviewIdInput.disabled = active;
    videoModeSelect.disabled = active;
    chatInput.disabled = !active;
    sendChatBtn.disabled = !active;
    hrIdInput.disabled = active; // Disable HR input during interview
    getSummaryBtn.disabled = active; // Disable summary button during interview

    if (!active) {
        mediaStatus.textContent = 'Interview stopped.';
        localVideo.srcObject = null;
        agentTextOutput.innerHTML = '';
        chatOutput.innerHTML = '';
        interviewSummary.innerHTML = '';
    } else {
        mediaStatus.textContent = 'Connecting...';
    }
}

async function startMedia() {
    const mode = videoModeSelect.value;
    let videoConstraints = false;
    let audioConstraints = { sampleRate: AUDIO_SAMPLE_RATE, echoCancellation: true }; // Request specific sample rate

    try {
        if (mode === 'camera') {
            videoConstraints = { facingMode: 'user' };
            mediaStatus.textContent = 'Requesting camera and microphone access...';
            localStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: audioConstraints });
        } else if (mode === 'screen') {
            mediaStatus.textContent = 'Requesting screen share and microphone access...';
            // Get display media first (screen/window)
            const screenStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false }); // Don't capture audio from screen share source

            // Then get user media for microphone only
            const audioStream = await navigator.mediaDevices.getUserMedia({ video: false, audio: audioConstraints });

            // Combine tracks
            localStream = new MediaStream([
                ...screenStream.getVideoTracks(),
                ...audioStream.getAudioTracks()
            ]);

            // Handle screen sharing stop event (e.g., user clicks "Stop sharing" button)
            screenStream.getVideoTracks()[0].onended = () => {
                logToChat('Screen sharing stopped by user.');
                stopInterview();
            };

        } else { // mode === 'none'
            mediaStatus.textContent = 'Requesting microphone access...';
            localStream = await navigator.mediaDevices.getUserMedia({ video: false, audio: audioConstraints });
        }

        logToChat(`Media access granted for mode: ${mode}.`);
        mediaStatus.textContent = 'Media stream active.';
        if (localStream.getVideoTracks().length > 0) {
            localVideo.srcObject = localStream;
            localVideo.style.display = 'block'; // Show video element
        } else {
             localVideo.style.display = 'none'; // Hide video element if no video track
        }

        return true;

    } catch (err) {
        console.error('Error accessing media devices.', err);
        mediaStatus.textContent = `Error: ${err.name} - ${err.message}`;
        logToChat(`Failed to get media: ${err.message}`, 'Error');
        localStream = null;
        return false;
    }
}

function connectWebSocket() {
    const interviewId = interviewIdInput.value.trim();
    const mode = videoModeSelect.value;

    if (!interviewId) {
        logToChat('Please enter an Interview ID.', 'Error');
        return false;
    }

    // Basic validation (similar to Python script logic)
    if (!interviewId.match(/^C\d{3,5}$/)) {
         logToChat('Invalid Interview ID format. Should be C followed by 3 to 5 digits (e.g., C123).', 'Error');
         return false;
    }


    logToChat('Attempting to connect to backend...');
    mediaStatus.textContent = 'Connecting to server...';

    // --- !!! BACKEND REQUIRED !!! ---
    // This part needs a running WebSocket server based on the Python logic.
    // The webSocket interactions below are placeholders.
    webSocket = new WebSocket(`${BACKEND_WS_URL}?interviewId=${interviewId}&mode=${mode}`); // Pass params

    webSocket.onopen = () => {
        logToChat('Connected to backend server.');
        mediaStatus.textContent = 'Connected. Starting interview...';
        // Send initial message or wait for backend prompt
        // webSocket.send(JSON.stringify({ type: 'start', interviewId: interviewId, mode: mode }));
        startAudioProcessing(); // Start sending audio after connection
    };

    webSocket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            console.log('Received message:', message);

            if (message.type === 'agent_text') {
                logAgentResponse(message.text);
            } else if (message.type === 'agent_audio') {
                // Placeholder: Need to decode base64 and play audio
                // This requires more complex audio handling (e.g., Web Audio API)
                playReceivedAudio(message.data); // Function to be implemented
                logToChat('Received audio chunk from agent.', 'System');
            } else if (message.type === 'chat') {
                 logToChat(message.text, message.sender || 'Agent');
            } else if (message.type === 'status') {
                 logToChat(message.text, 'System');
                 mediaStatus.textContent = message.text;
            } else if (message.type === 'summary') {
                interviewSummary.innerHTML = `<pre>${message.text}</pre>`; // Display summary
            } else if (message.type === 'error') {
                logToChat(message.text, 'Error');
                mediaStatus.textContent = `Error: ${message.text}`;
            }
            // Add more message types as needed (e.g., validation results)

        } catch (error) {
            console.error('Failed to parse message or invalid message format:', event.data, error);
            logToChat('Received unparseable message from backend.', 'Error');
        }
    };

    webSocket.onerror = (error) => {
        console.error('WebSocket Error:', error);
        logToChat('WebSocket connection error. Check if the backend server is running.', 'Error');
        mediaStatus.textContent = 'Connection error.';
        setUIState(false); // Reset UI on error
    };

    webSocket.onclose = (event) => {
        logToChat(`WebSocket connection closed: ${event.reason} (Code: ${event.code})`, 'System');
        mediaStatus.textContent = 'Disconnected.';
        stopAudioProcessing();
        if (isInterviewActive) { // Avoid resetting if stopped manually
            setUIState(false);
        }
    };

    return true; // Indicate connection attempt started
}

// --- Web Audio API for processing and sending audio ---
async function startAudioProcessing() {
    if (!localStream || localStream.getAudioTracks().length === 0) {
        logToChat('No microphone stream available to process.', 'Error');
        return;
    }

    if (!webSocket || webSocket.readyState !== WebSocket.OPEN) {
        logToChat('WebSocket not connected. Cannot send audio.', 'Error');
        return;
    }

    // Create AudioContext if it doesn't exist
    if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: AUDIO_SAMPLE_RATE // Ensure context matches desired rate
        });
        console.info(`AudioContext created with sample rate: ${audioContext.sampleRate}`);
    }

    // Get the audio track from the local stream
    const audioTracks = localStream.getAudioTracks();
    if (audioTracks.length === 0) {
        logToChat('No audio tracks found in the local stream.', 'Error');
        return;
    }
    const audioTrack = audioTracks[0];
    console.info(`Using audio track: ${audioTrack.label} (ID: ${audioTrack.id})`);
    console.info('Audio track settings:', audioTrack.getSettings());


    // Use MediaRecorder for simpler chunking
    // Try specific MIME types with fallbacks
    const preferredMimeTypes = [
        'audio/webm;codecs=opus', // High quality, widely supported
        'audio/webm',             // WebM audio
        'audio/ogg;codecs=opus',  // Ogg Opus
        'audio/ogg',              // Ogg Vorbis
        'audio/wav',              // WAV (less efficient)
    ];

    let selectedMimeType = null;

    for (const mimeType of preferredMimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
            selectedMimeType = mimeType;
            console.info(`Using supported MIME type: ${mimeType}`);
            break;
        }
    }

    // Create a new MediaStream with only the selected audio track for the MediaRecorder
    const audioStreamForRecorder = new MediaStream([audioTrack]);

    if (!selectedMimeType) {
        console.warn("No preferred MIME types supported, trying default.");
        try {
             mediaRecorder = new MediaRecorder(audioStreamForRecorder); // Use stream with only audio track
             selectedMimeType = mediaRecorder.mimeType;
             console.info(`Using browser default MIME type: ${selectedMimeType}`);
        } catch (e) {
             console.error("MediaRecorder failed completely with default:", e);
             logToChat("Failed to create MediaRecorder for audio.", "Error");
             return;
        }
    } else {
        try {
            mediaRecorder = new MediaRecorder(audioStreamForRecorder, { mimeType: selectedMimeType }); // Use stream with only audio track
        } catch (e) {
            console.error(`MediaRecorder failed with ${selectedMimeType}:`, e);
            logToChat(`Failed to create MediaRecorder with ${selectedMimeType}.`, "Error");
             try {
                 mediaRecorder = new MediaRecorder(audioStreamForRecorder); // Fallback with stream and default
                 selectedMimeType = mediaRecorder.mimeType;
                 console.info(`Falling back to browser default MIME type: ${selectedMimeType}`);
             } catch (e2) {
                 console.error("MediaRecorder failed completely with fallback:", e2);
                 logToChat("Failed to create MediaRecorder for audio.", "Error");
                 return;
             }
        }
    }

    if (!mediaRecorder) {
         logToChat("Failed to initialize MediaRecorder.", "Error");
         return;
    }

    // Update the backend message sending to include the actual mimeType used
    // This requires a change in the backend to handle different mime types if necessary.
    // For now, we assume the backend can handle 'audio/webm'. If other types are used,
    //# the backend's audio processing logic might need adjustment.
    const audioMimeTypeForBackend = selectedMimeType.split(';')[0]; // Send just the base type


    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && webSocket && webSocket.readyState === WebSocket.OPEN) {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64Audio = reader.result.split(',')[1];
                 if (webSocket && webSocket.readyState === WebSocket.OPEN) {
                    // Include the mimeType used for this chunk
                    webSocket.send(JSON.stringify({ type: 'audio_chunk', data: base64Audio, mimeType: audioMimeTypeForBackend }));
                 }
            };
            reader.readAsDataURL(event.data);
        }
    };

    mediaRecorder.onstart = () => {
        logToChat(`Started recording audio chunks using ${selectedMimeType}.`, 'System');
    };

     mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event.error);
        logToChat(`MediaRecorder error: ${event.error.name}`, 'Error');
    };

    mediaRecorder.start(AUDIO_CHUNK_DURATION_MS); // Collect chunks every X ms

    logToChat('Audio processing started.', 'System');
}


function stopAudioProcessing() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        logToChat('Stopped recording audio chunks.', 'System');
    }
     if (audioContext && audioContext.state === 'running') { // Check if running before closing
        audioContext.close().then(() => logToChat('AudioContext closed.', 'System'));
    } else if (audioContext) {
         logToChat('AudioContext is not in running state, skipping close.', 'System');
    }
    mediaRecorder = null;
    audioContext = null;
}

// Placeholder for playing received audio
async function playReceivedAudio(base64Audio) {
    // This needs a proper implementation using Web Audio API
    // 1. Decode base64 string to ArrayBuffer
    // 2. Decode ArrayBuffer to AudioBuffer using audioContext.decodeAudioData
    // 3. Create an AudioBufferSourceNode
    // 4. Set its buffer to the decoded AudioBuffer
    // 5. Connect the source node to audioContext.destination
    // 6. Start the source node (.start(0))
    console.log("Placeholder: Would play received audio chunk now.");
    // Example structure (needs error handling and proper context management):
    /*
    if (!audioContext) audioContext = new AudioContext({ sampleRate: RECEIVE_SAMPLE_RATE }); // Use appropriate sample rate
    const audioData = Uint8Array.from(atob(base64Audio), c => c.charCodeAt(0)).buffer;
    audioContext.decodeAudioData(audioData, (buffer) => {
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);
        source.start(0);
    }, (error) => {
        console.error('Error decoding audio data:', error);
        logToChat('Error playing agent audio.', 'Error');
    });
    */
}


async function startInterview() {
    setUIState(true); // Disable controls, enable stop button etc.
    logToChat('Starting interview process...');

    const mediaStarted = await startMedia();
    if (!mediaStarted) {
        logToChat('Failed to start media. Aborting interview.', 'Error');
        setUIState(false);
        return;
    }

    const connected = connectWebSocket();
    if (!connected) {
        logToChat('Failed to initiate connection. Aborting interview.', 'Error');
        // Clean up media stream if connection fails immediately
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
            localVideo.srcObject = null;
        }
        setUIState(false);
    }
    // If connected, WebSocket 'onopen' will handle the next steps (like starting audio processing)
}

function stopInterview() {
    logToChat('Stopping interview...');
    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
        webSocket.send(JSON.stringify({ type: 'stop' })); // Notify backend
        webSocket.close(1000, 'User stopped interview'); // Close WebSocket gracefully
    }
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        logToChat('Media streams stopped.');
    }
    stopAudioProcessing();

    localStream = null;
    webSocket = null;
    setUIState(false);
}

function sendChatMessage() {
    const text = chatInput.value.trim();
    if (text && webSocket && webSocket.readyState === WebSocket.OPEN) {
        logToChat(text, 'You'); // Log user's message immediately
        webSocket.send(JSON.stringify({ type: 'chat', text: text }));
        chatInput.value = ''; // Clear input field
    } else if (!text) {
         logToChat('Cannot send empty message.', 'System');
    } else {
        logToChat('Not connected to the server. Cannot send message.', 'Error');
    }
}

function requestSummary() {
    const hrId = hrIdInput.value.trim();
    if (!hrId) {
        logToChat('Please enter an HR ID to request the summary.', 'Error');
        interviewSummary.textContent = 'HR ID required.';
        return;
    }
     // Basic validation
    if (!hrId.match(/^H\d+/)) { // Starts with H, followed by digits
         logToChat('Invalid HR ID format. Should start with H followed by digits (e.g., H1234).', 'Error');
         interviewSummary.textContent = 'Invalid HR ID format.';
         return;
    }

    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
         logToChat(`HR (${hrId}) is requesting summary... (This message should ideally only go to backend)`, 'System');
         webSocket.send(JSON.stringify({ type: 'request_summary', hrId: hrId }));
         interviewSummary.textContent = 'Requesting summary from agent...';
    } else {
         // Allow requesting summary even if disconnected, assuming backend stores it
         // This requires a different mechanism (e.g., HTTP request) if WS is closed
         logToChat('WebSocket not connected. Cannot request summary via WebSocket. (Needs alternative method)', 'Error');
         interviewSummary.textContent = 'Cannot request summary - not connected.';
         // TODO: Implement HTTP fallback if needed
    }
}


// --- Event Listeners ---
startInterviewBtn.addEventListener('click', startInterview);
stopInterviewBtn.addEventListener('click', stopInterview);

sendChatBtn.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
});

getSummaryBtn.addEventListener('click', requestSummary);

// --- Initial Setup ---
setUIState(false); // Initial state is inactive

// Optional: Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (isInterviewActive) {
        stopInterview(); // Attempt to clean up if user navigates away
    }
});
