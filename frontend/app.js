document.addEventListener('DOMContentLoaded', function() {
    let video = document.getElementById("video");
    let canvas = document.getElementById("canvas");
    let ctx = canvas.getContext("2d");
    let cameraSelect = document.getElementById("cameraSelect");
    let startButton = document.getElementById("startButton");
    let stopButton = document.getElementById("stopButton");
    let statusText = document.getElementById("status");
    let cameraStream = null;
    let hands = null;
    let isModelLoaded = false;
    let animationId = null;
    let lastResults = null;
    let lastProcessedTime = 0;
    let processingFrame = false;

    // Immediately hide the video element when the page loads
    video.style.display = "none";
    video.style.position = "absolute";
    video.style.left = "-9999px";

    // Processing frequency (ms) - only process frames at this interval
    const PROCESSING_INTERVAL = 100; // 10 fps for hand detection

    // Set fixed dimensions for processing
    const CANVAS_WIDTH = 640;
    const CANVAS_HEIGHT = 480;
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;

    // Define colors for each finger and palm
    const landmarkColors = {
        0: '#0000FF',  // Blue - Wrist
        1: '#0000FF',  // Blue - Thumb base (CMC joint)
        2: '#FFFFFF',  // White - Thumb
        3: '#FFFFFF',  // White - Thumb
        4: '#FFFFFF',  // White - Thumb tip
        5: '#0000FF',  // Blue - Index finger base (MCP joint)
        6: '#9400D3',  // Purple - Index finger
        7: '#9400D3',  // Purple - Index finger
        8: '#9400D3',  // Purple - Index finger tip
        9: '#0000FF',  // Blue - Middle finger base (MCP joint)
        10: '#00FFFF', // Cyan - Middle finger
        11: '#00FFFF', // Cyan - Middle finger
        12: '#00FFFF', // Cyan - Middle finger tip
        13: '#0000FF', // Blue - Ring finger base (MCP joint)
        14: '#00FF00', // Green - Ring finger
        15: '#00FF00', // Green - Ring finger
        16: '#00FF00', // Green - Ring finger tip
        17: '#0000FF', // Blue - Pinky base (MCP joint)
        18: '#FF7F00', // Orange - Pinky
        19: '#FF7F00', // Orange - Pinky
        20: '#FF7F00'  // Orange - Pinky tip
    };

    // Define colors for connections
    const connectionColors = {
        // Palm connections (all grey)
        "5-9": '#808080',   // Index base to Middle base
        "9-13": '#808080',  // Middle base to Ring base
        "13-17": '#808080', // Ring base to Pinky base
        "5-0": '#808080',   // Index base to Wrist
        "17-0": '#808080',  // Pinky base to Wrist
        "1-0": '#808080',   // Thumb base to Wrist
        
        // Thumb connections (white)
        "1-2": '#FFFFFF',
        "2-3": '#FFFFFF',
        "3-4": '#FFFFFF',
        
        // Index finger connections (purple)
        "5-6": '#9400D3',
        "6-7": '#9400D3',
        "7-8": '#9400D3',
        
        // Middle finger connections (cyan)
        "9-10": '#00FFFF',
        "10-11": '#00FFFF',
        "11-12": '#00FFFF',
        
        // Ring finger connections (green)
        "13-14": '#00FF00',
        "14-15": '#00FF00',
        "15-16": '#00FF00',
        
        // Pinky connections (orange)
        "17-18": '#FF7F00',
        "18-19": '#FF7F00',
        "19-20": '#FF7F00'
    };

    // Define custom connections to match the requested pattern
    const customConnections = [
        // Palm connections - updated to connect side to side and then down
        [5, 9],    // Index base to Middle base
        [9, 13],   // Middle base to Ring base
        [13, 17],  // Ring base to Pinky base
        [5, 0],    // Index base to Wrist
        [17, 0],   // Pinky base to Wrist
        [1, 0],    // Thumb base to Wrist
        
        // Thumb connections
        [1, 2], [2, 3], [3, 4],
        
        // Index finger connections
        [5, 6], [6, 7], [7, 8],
        
        // Middle finger connections
        [9, 10], [10, 11], [11, 12],
        
        // Ring finger connections
        [13, 14], [14, 15], [15, 16],
        
        // Pinky connections
        [17, 18], [18, 19], [19, 20]
    ];

    // Custom function to draw connections with specific colors
    function drawCustomConnections(ctx, landmarks, connections) {
        if (!landmarks || !connections) return;
        
        connections.forEach(connection => {
            const [start, end] = connection;
            const startPoint = landmarks[start];
            const endPoint = landmarks[end];
            
            // Get connection key
            const connectionKey = `${start}-${end}`;
            // Use the defined color or fall back to blue
            const color = connectionColors[connectionKey] || '#0000FF';
            
            // Draw the connection
            ctx.beginPath();
            ctx.moveTo(startPoint.x * canvas.width, startPoint.y * canvas.height);
            ctx.lineTo(endPoint.x * canvas.width, endPoint.y * canvas.height);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2; // Connection thickness set to 2
            ctx.stroke();
        });
    }

    // Main loop that handles both rendering and processing
    function mainLoop(timestamp) {
        // First, render the current frame regardless of processing
        renderFrame();
        
        // Check if it's time to process a new frame
        if (timestamp - lastProcessedTime >= PROCESSING_INTERVAL && !processingFrame) {
            processFrame();
            lastProcessedTime = timestamp;
        }
        
        // Continue the loop if stream is active
        if (cameraStream && cameraStream.active) {
            animationId = requestAnimationFrame(mainLoop);
        }
    }
    
    // Function to render the current frame
    function renderFrame() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // If video is ready, draw it first
        if (video.readyState === 4) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }
        
        // If we have hand detection results, draw them
        if (lastResults && lastResults.multiHandLandmarks && lastResults.multiHandLandmarks.length > 0) {
            const landmarks = lastResults.multiHandLandmarks[0];
            
            // Draw hand landmarks and connections
            drawCustomConnections(ctx, landmarks, customConnections);
            
            // Draw the landmark points with custom colors
            for (let i = 0; i < landmarks.length; i++) {
                const landmark = landmarks[i];
                const color = landmarkColors[i] || '#FFFFFF';
                
                ctx.beginPath();
                ctx.arc(
                    landmark.x * canvas.width,
                    landmark.y * canvas.height,
                    5, // Landmark thickness set to 5
                    0,
                    2 * Math.PI
                );
                ctx.fillStyle = color;
                ctx.fill();
            }
            
            // Draw bounding box
            if (lastResults.prediction) {
                drawBoundingBox(ctx, landmarks, lastResults.prediction.prediction_class);
            }
        }
    }
    
    // Function to process a frame through MediaPipe and get prediction
    async function processFrame() {
        if (!isModelLoaded || !cameraStream || !hands || video.readyState !== 4) {
            return;
        }
        
        processingFrame = true;
        
        try {
            // Send the video frame to MediaPipe
            await hands.send({ image: video });
        } catch (error) {
            console.error("Error during hand detection:", error);
            statusText.innerText = "Detection error: " + error.message;
        } finally {
            processingFrame = false;
        }
    }

    async function getCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === "videoinput");

            cameraSelect.innerHTML = "";
            videoDevices.forEach((device, index) => {
                const option = document.createElement("option");
                option.value = device.deviceId;
                option.text = device.label || `Camera ${index + 1}`;
                cameraSelect.appendChild(option);
            });

            if (videoDevices.length > 1) {
                cameraSelect.selectedIndex = 1;  // Default to secondary camera
            }
        } catch (error) {
            console.error("Error getting cameras:", error);
            statusText.innerText = "Error accessing cameras: " + error.message;
        }
    }

    async function startCamera(deviceId) {
        try {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
            }
            
            // Set specific constraints including resolution
            const constraints = { 
                video: { 
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: CANVAS_WIDTH },
                    height: { ideal: CANVAS_HEIGHT }
                } 
            };
            
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = cameraStream;
            
            // Wait for video to be ready without showing it
            await new Promise(resolve => {
                video.onloadedmetadata = () => {
                    video.play();
                    resolve();
                };
            });
            
            startButton.style.display = "none";
            stopButton.style.display = "inline";
            statusText.innerText = "Camera Started! Waiting for hand detection...";

            // Start the main loop
            lastProcessedTime = 0;
            requestAnimationFrame(mainLoop);
        } catch (error) {
            console.error("Error starting camera:", error);
            statusText.innerText = "Error starting camera: " + error.message;
        }
    }

    function stopCamera() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        
        video.srcObject = null;
        lastResults = null;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        startButton.style.display = "inline";
        stopButton.style.display = "none";
        statusText.innerText = "Camera Stopped.";
    }

    async function loadMediaPipe() {
        try {
            console.log("Loading MediaPipe...");
            
            // Load the required scripts
            await import('https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/hands.js');
            await import('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1620248257/drawing_utils.js');
            
            // Access the global objects created by the imports
            const handsModule = window.Hands;
            
            if (!handsModule) {
                throw new Error("MediaPipe Hands module not found on window object");
            }
            
            hands = new handsModule({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`
            });

            hands.setOptions({
                maxNumHands: 1, // Limit to 1 hand for better performance
                modelComplexity: 0, // Use simpler model (0 instead of 1)
                minDetectionConfidence: 0.6,
                minTrackingConfidence: 0.5
            });

            hands.onResults(async results => {
                if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                    try {
                        const landmarks = results.multiHandLandmarks[0];
                        const prediction = await getPrediction(landmarks);
                        
                        // Store results with prediction for rendering
                        lastResults = { ...results, prediction };
                        
                        statusText.innerText = "Hand detected";

                    } catch (error) {
                        console.error("Error processing hand data:", error);
                        statusText.innerText = "Processing error: " + error.message;
                    }
                } else {
                    // If no hands are detected, clear the last results to remove the visualization
                    if (lastResults && lastResults.multiHandLandmarks && lastResults.multiHandLandmarks.length > 0) {
                        lastResults = { multiHandLandmarks: [] };
                        statusText.innerText = "No hands detected";
                    }
                }
            });

            isModelLoaded = true;
            statusText.innerText = "MediaPipe Loaded! Click 'Start Camera' to begin.";
            console.log("MediaPipe loaded successfully.");

        } catch (error) {
            console.error("MediaPipe failed to load:", error);
            statusText.innerText = "Error loading MediaPipe: " + error.message;
        }
    }

    // Function to draw a bounding box around the hand and display the prediction
    function drawBoundingBox(ctx, landmarks, prediction) {
        const xCoords = landmarks.map(l => l.x * canvas.width);
        const yCoords = landmarks.map(l => l.y * canvas.height);
        
        const minX = Math.min(...xCoords);
        const minY = Math.min(...yCoords);
        const maxX = Math.max(...xCoords);
        const maxY = Math.max(...yCoords);
        
        // Draw the bounding box
        ctx.strokeStyle = '#FF0000'; // Red color for the bounding box
        ctx.lineWidth = 2; // Line width for the bounding box
        ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
    
        // Draw the prediction text with background for better visibility
        const textX = (minX + maxX) / 2;
        const textY = minY - 10;
        
        // Include confidence in the display text if available
        let text = prediction;
        if (lastResults && lastResults.prediction && lastResults.prediction.confidence) {
            const confidence = (lastResults.prediction.confidence * 100).toFixed(1);
            text = `${prediction} (${confidence}%)`;
        }
        
        // Measure text width for background
        ctx.font = '16px Arial';
        const metrics = ctx.measureText(text);
        const textWidth = metrics.width;
        
        // Draw text background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(textX - textWidth/2 - 5, textY - 16, textWidth + 10, 20);
        
        // Draw text
        ctx.fillStyle = '#FFFFFF';
        ctx.textAlign = 'center';
        ctx.fillText(text, textX, textY);
    }

    // Function to get the prediction from the API with debouncing
    const getPrediction = (function() {
        let lastRequestTime = 0;
        const DEBOUNCE_TIME = 300; // ms
        
        return async function(landmarks) {
            const now = Date.now();
            
            // Skip if we're calling too frequently
            if (now - lastRequestTime < DEBOUNCE_TIME) {
                // Return the last prediction if available
                if (lastResults && lastResults.prediction) {
                    return lastResults.prediction;
                }
                
                // Return a placeholder if no previous prediction
                return {
                    prediction_class: "Processing...",
                    confidence: 0,
                    error: false
                };
            }
            
            lastRequestTime = now;
            
            // Prepare the data in the expected format (21x3)
            const landmarkArray = landmarks.map(landmark => [landmark.x, landmark.y, landmark.z]);
        
            try {
                const response = await fetch(ASL_API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-KEY': JS_KEY
                    },
                    body: JSON.stringify({ landmarks: landmarkArray }),
                });
        
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
        
                const data = await response.json();
                
                // Add default fallback if prediction_class is not present
                return {
                    prediction_class: data.prediction_class || 'Unknown',
                    confidence: data.confidence || 0,
                    error: !data.prediction_class
                };
            } catch (error) {
                console.error("Error fetching prediction:", error);
                return {
                    prediction_class: 'Error',
                    confidence: 0,
                    error: true
                };
            }
        };
    })();

    startButton.addEventListener("click", () => startCamera(cameraSelect.value));
    stopButton.addEventListener("click", stopCamera);

    async function init() {
        try {
            // Check for camera access
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            // If successful, stop the stream immediately
            stream.getTracks().forEach(track => track.stop());
            
            await getCameras();
            await loadMediaPipe();
        } catch (error) {
            console.error("Camera access denied or not available:", error);
            statusText.innerText = "Camera access is required for this application. Please allow camera access.";
        }
    }

    init();
});
