import React, { useRef, useEffect, useState, Suspense } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, POSE_CONNECTIONS, HAND_CONNECTIONS } from "@mediapipe/drawing_utils";

// 3D Imports
import { Canvas } from "@react-three/fiber";
import FloatingText from "./FloatingText";

// Alphabetical List (Must match Python training EXACTLY)
const CLASS_LABELS = ['america', 'book', 'christmas', 'confused', 'earth', 'help', 'intelligent', 'love', 'music', 'pain', 'person', 'sick', 'stop', 'technology', 'welcome'];
const CameraComponent = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState("Loading...");
  const [confidence, setConfidence] = useState(0);

  // Logic Buffers
  const predictionBuffer = useRef([]); 
  const frameCounter = useRef(0); // For throttling

  useEffect(() => {
    const loadModel = async () => {
      try {
        // Load the JSON model from the public folder
        const net = await tf.loadLayersModel("/model/model.json");
        setModel(net);
        setPrediction("Tacit OS");
      } catch (err) {
        console.error(err);
        setPrediction("System Failure");
      }
    };
    loadModel();
  }, []);

  const extractKeypoints = (results) => {
    // Pose: 33 * 4 = 132
    const pose = results.poseLandmarks 
      ? results.poseLandmarks.flatMap(res => [res.x, res.y, res.z, res.visibility]) 
      : new Array(132).fill(0);

    // Left Hand: 21 * 3 = 63
    const lh = results.leftHandLandmarks 
      ? results.leftHandLandmarks.flatMap(res => [res.x, res.y, res.z]) 
      : new Array(63).fill(0);

    // Right Hand: 21 * 3 = 63
    const rh = results.rightHandLandmarks 
      ? results.rightHandLandmarks.flatMap(res => [res.x, res.y, res.z]) 
      : new Array(63).fill(0);

    return [...pose, ...lh, ...rh];
  };

  useEffect(() => {
    let sequence = []; 

    const onResults = async (results) => {
      if (!canvasRef.current || !webcamRef.current || !webcamRef.current.video) return;

      // 1. Draw Skeleton (Visual Feedback)
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
      
      const ctx = canvasRef.current.getContext("2d");
      ctx.save();
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Draw connectors (Green for body, White for hands)
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "rgba(0, 255, 0, 0.5)", lineWidth: 1 });
      drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "rgba(255, 255, 255, 0.8)", lineWidth: 1 });
      drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "rgba(255, 255, 255, 0.8)", lineWidth: 1 });
      ctx.restore();

      // 2. Data Collection (Run every frame to fill buffer)
      if (model) {
        const keypoints = extractKeypoints(results);
        sequence.push(keypoints);
        if (sequence.length > 30) sequence.shift(); // Keep only last 30 frames

        // 3. AI Inference (Run ONLY every 3rd frame to save CPU)
        frameCounter.current += 1;
        if (sequence.length === 30 && frameCounter.current % 3 === 0) {
          
          const input = tf.tensor([sequence]); 
          const output = model.predict(input);
          const values = await output.data();
          const maxIndex = values.indexOf(Math.max(...values));
          const currentWord = CLASS_LABELS[maxIndex];
          const currentConf = values[maxIndex];

          input.dispose(); output.dispose(); // Clean up memory

          // --- FORGIVING SMOOTHING LOGIC ---
          // Threshold reduced to 60% so it's less "shy"
          if (currentConf > 0.60) {
            
            predictionBuffer.current.push(currentWord);
            
            // Only look at last 7 predictions (approx 0.5 seconds)
            if (predictionBuffer.current.length > 7) {
              predictionBuffer.current.shift();
            }
            
            // Majority Vote: If 4+ frames agree, show the word
            const counts = {};
            predictionBuffer.current.forEach(word => { counts[word] = (counts[word] || 0) + 1; });
            
            const mostFrequentWord = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
            
            if (counts[mostFrequentWord] >= 4) {
              setPrediction(mostFrequentWord.toUpperCase());
              setConfidence(Math.round(currentConf * 100));
            }

          } else {
             // If confidence is low, clear buffer (Reset)
             predictionBuffer.current = [];
          }
        }
      }
    };

    // Initialize MediaPipe
    const holistic = new Holistic({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}` });
    holistic.setOptions({ 
      modelComplexity: 1, 
      smoothLandmarks: true, 
      minDetectionConfidence: 0.5, 
      minTrackingConfidence: 0.5 
    });
    holistic.onResults(onResults);

    if (webcamRef.current) {
      const camera = new Camera(webcamRef.current.video, {
        onFrame: async () => { await holistic.send({ image: webcamRef.current.video }); },
        width: 640, height: 480
      });
      camera.start();
    }
  }, [model]);

  return (
    <div style={{ width: "100vw", height: "100vh", backgroundColor: "#111", overflow: "hidden", position: "relative" }}>
      
      {/* --- LAYER 1: THE GAME WORLD (3D Text Center) --- */}
      <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", zIndex: 1 }}>
        <Canvas camera={{ position: [0, 0, 6], fov: 60 }}>
          <ambientLight intensity={0.6} />
          <spotLight position={[10, 10, 10]} angle={0.3} penumbra={1} intensity={1} />
          <pointLight position={[-10, -5, -10]} color="#00FF00" intensity={0.5} />
          
          <Suspense fallback={null}>
            <FloatingText text={prediction} />
          </Suspense>
        </Canvas>
      </div>

      {/* --- LAYER 2: THE HUD (Webcam + Confidence) --- */}
      <div style={{
        position: "absolute",
        bottom: "30px",
        right: "30px",
        width: "320px",
        height: "240px",
        borderRadius: "15px",
        overflow: "hidden",
        border: "2px solid #333",
        boxShadow: "0px 10px 30px rgba(0,0,0,0.8)",
        zIndex: 10,
        backgroundColor: "black"
      }}>
        {/* Webcam Video */}
        <Webcam 
          ref={webcamRef} 
          style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }} 
        />
        
        {/* Skeleton Overlay */}
        <canvas 
          ref={canvasRef} 
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", transform: "scaleX(-1)" }} 
        />

        {/* Confidence Badge */}
        <div style={{
          position: "absolute",
          bottom: "10px",
          left: "10px",
          backgroundColor: "rgba(0, 0, 0, 0.7)",
          color: confidence > 80 ? "#00FF00" : (confidence > 50 ? "#FFFF00" : "#FF0000"),
          padding: "4px 8px",
          borderRadius: "4px",
          fontSize: "12px",
          fontWeight: "bold",
          fontFamily: "monospace"
        }}>
          ACCURACY: {confidence}%
        </div>
      </div>

    </div>
  );
};

export default CameraComponent;