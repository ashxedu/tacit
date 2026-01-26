import React, { useRef, useEffect, useState, Suspense } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks, POSE_CONNECTIONS, HAND_CONNECTIONS } from "@mediapipe/drawing_utils";

// 3D Imports
import { Canvas } from "@react-three/fiber";
import FloatingText from "./FloatingText";

// Alphabetical List
const CLASS_LABELS = [
  'go', 'goodbye', 'happy', 'hello', 'help', 
  'intelligent', 'love', 'me', 'no', 'please', 
  'sad', 'sorry', 'stop', 'yes', 'you'
];

const CameraComponent = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState("Loading...");
  const [confidence, setConfidence] = useState(0);

  // Smoothing Buffer
  const predictionBuffer = useRef([]); 

  useEffect(() => {
    const loadModel = async () => {
      try {
        const net = await tf.loadLayersModel("/model/model.json");
        setModel(net);
        setPrediction("Tacit OS");
      } catch (err) {
        setPrediction("System Failure");
      }
    };
    loadModel();
  }, []);

  const extractKeypoints = (results) => {
    const pose = results.poseLandmarks ? results.poseLandmarks.flatMap(res => [res.x, res.y, res.z, res.visibility]) : new Array(33 * 4).fill(0);
    const lh = results.leftHandLandmarks ? results.leftHandLandmarks.flatMap(res => [res.x, res.y, res.z]) : new Array(21 * 3).fill(0);
    const rh = results.rightHandLandmarks ? results.rightHandLandmarks.flatMap(res => [res.x, res.y, res.z]) : new Array(21 * 3).fill(0);
    return [...pose, ...lh, ...rh];
  };

  useEffect(() => {
    let sequence = []; 

    const onResults = async (results) => {
      if (!canvasRef.current || !webcamRef.current || !webcamRef.current.video) return;

      // 1. Set Canvas Dimensions to match Video (Internal Resolution)
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
      
      const ctx = canvasRef.current.getContext("2d");
      ctx.save();
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // 2. Draw Skeleton (Thinner lines for the small view)
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "rgba(0, 255, 0, 0.5)", lineWidth: 1 });
      drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "rgba(255, 255, 255, 0.8)", lineWidth: 1 });
      drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "rgba(255, 255, 255, 0.8)", lineWidth: 1 });
      ctx.restore();

      // 3. AI Inference
      if (model) {
        const keypoints = extractKeypoints(results);
        sequence.push(keypoints);
        if (sequence.length > 30) sequence.shift();

        if (sequence.length === 30) {
          const input = tf.tensor([sequence]); 
          const output = model.predict(input);
          const values = await output.data();
          const maxIndex = values.indexOf(Math.max(...values));
          const currentWord = CLASS_LABELS[maxIndex];
          const currentConf = values[maxIndex];

          input.dispose(); output.dispose();

          // Strict Smoothing Logic
          if (currentConf > 0.85) {
            predictionBuffer.current.push(currentWord);
            if (predictionBuffer.current.length > 15) predictionBuffer.current.shift();
            
            const isStable = predictionBuffer.current.every(word => word === currentWord);
            if (isStable && predictionBuffer.current.length === 15) {
              setPrediction(currentWord.toUpperCase());
              setConfidence(Math.round(currentConf * 100));
            }
          } else {
             predictionBuffer.current = [];
          }
        }
      }
    };

    const holistic = new Holistic({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}` });
    holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });
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
        width: "320px",  // Mini-map size
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
          style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }} // Mirror effect
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
          color: confidence > 80 ? "#00FF00" : "#888",
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