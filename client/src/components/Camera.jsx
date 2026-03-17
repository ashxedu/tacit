import React, { useRef, useEffect, useState, Suspense } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, POSE_CONNECTIONS, HAND_CONNECTIONS } from "@mediapipe/drawing_utils";

import { Canvas } from "@react-three/fiber";
import FloatingText from "./FloatingText";

const CameraComponent = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState("Loading...");
  const [confidence, setConfidence] = useState(0);

  const predictionBuffer = useRef([]); 
  const frameCounter = useRef(0); 
  const classLabelsRef = useRef([]); 

  useEffect(() => {
    const loadModelAndClasses = async () => {
      try {
        const net = await tf.loadLayersModel("/model/model.json");
        setModel(net);
        
        const response = await fetch("/model/classes.json");
        classLabelsRef.current = await response.json();
        
        setPrediction("Tacit OS");
      } catch (err) {
        console.error("Error loading model or classes:", err);
        setPrediction("System Failure");
      }
    };
    loadModelAndClasses();
  }, []);

  const extractKeypoints = (results) => {
    const pose = results.poseLandmarks 
      ? results.poseLandmarks.flatMap(res => [res.x, res.y, res.z, res.visibility]) 
      : new Array(132).fill(0);

    const lh = results.leftHandLandmarks 
      ? results.leftHandLandmarks.flatMap(res => [res.x, res.y, res.z]) 
      : new Array(63).fill(0);

    const rh = results.rightHandLandmarks 
      ? results.rightHandLandmarks.flatMap(res => [res.x, res.y, res.z]) 
      : new Array(63).fill(0);

    return [...pose, ...lh, ...rh];
  };

  useEffect(() => {
    let sequence = []; 

    // --- NEW GEOMETRY FILTER ---
    const normalizeAndPrune = (frame) => {
      // 1. Prune lower body (Keep only first 25 pose landmarks: 25 * 4 = 100 features)
      const posePruned = frame.slice(0, 100);
      const lh = frame.slice(132, 195);
      const rh = frame.slice(195, 258);

      // 2. Find Anchor (Midpoint of left shoulder [index 11] and right shoulder [index 12])
      // 11 * 4 = 44 | 12 * 4 = 48
      const ls = [frame[44], frame[45], frame[46]];
      const rs = [frame[48], frame[49], frame[50]];

      const anchor = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, (ls[2] + rs[2]) / 2];

      // 3. Create the Shoulder Ruler (Scale Invariance)
      let shoulderDist = Math.hypot(ls[0] - rs[0], ls[1] - rs[1], ls[2] - rs[2]);
      if (shoulderDist < 1e-5) shoulderDist = 1.0; // Crash prevention

      const prunedFrame = [...posePruned, ...lh, ...rh]; // Length is now 226

      // 4. Normalize Pose (x, y, z) - Skip visibility
      for (let i = 0; i < 100; i += 4) {
        if (prunedFrame[i] === 0 && prunedFrame[i+1] === 0) continue;
        prunedFrame[i] = (prunedFrame[i] - anchor[0]) / shoulderDist;
        prunedFrame[i+1] = (prunedFrame[i+1] - anchor[1]) / shoulderDist;
        prunedFrame[i+2] = (prunedFrame[i+2] - anchor[2]) / shoulderDist;
      }

      // 5. Normalize Hands (x, y, z)
      for (let i = 100; i < 226; i += 3) {
        if (prunedFrame[i] === 0 && prunedFrame[i+1] === 0) continue;
        prunedFrame[i] = (prunedFrame[i] - anchor[0]) / shoulderDist;
        prunedFrame[i+1] = (prunedFrame[i+1] - anchor[1]) / shoulderDist;
        prunedFrame[i+2] = (prunedFrame[i+2] - anchor[2]) / shoulderDist;
      }

      return prunedFrame;
    };

    const onResults = async (results) => {
      if (!canvasRef.current || !webcamRef.current || !webcamRef.current.video) return;

      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
      
      const ctx = canvasRef.current.getContext("2d");
      ctx.save();
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "rgba(0, 255, 0, 0.5)", lineWidth: 1 });
      drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "rgba(255, 255, 255, 0.8)", lineWidth: 1 });
      drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "rgba(255, 255, 255, 0.8)", lineWidth: 1 });
      ctx.restore();

      if (model && classLabelsRef.current.length > 0) {
        const keypoints = extractKeypoints(results);
        
        // Apply the geometry math before pushing to the buffer
        const normalizedKeypoints = normalizeAndPrune(keypoints); 
        sequence.push(normalizedKeypoints);
        
        if (sequence.length > 30) sequence.shift(); 

        frameCounter.current += 1;
        if (sequence.length === 30 && frameCounter.current % 3 === 0) {
          
          // --- THE FRONTEND VELOCITY HACK ---
          const sequenceWithVelocity = sequence.map((frame, i) => {
            if (i === 0) {
              // Now padding with 226 zeros instead of 258
              return [...frame, ...new Array(226).fill(0)]; 
            }
            const prevFrame = sequence[i - 1];
            const velocity = frame.map((val, j) => val - prevFrame[j]);
            return [...frame, ...velocity];
          });

          const input = tf.tensor([sequenceWithVelocity]); 
          const output = model.predict(input);
          const values = await output.data();
          const maxIndex = values.indexOf(Math.max(...values));
          
          const currentWord = classLabelsRef.current[maxIndex];
          const currentConf = values[maxIndex];

          input.dispose(); output.dispose(); 

          if (currentConf > 0.60) {
            predictionBuffer.current.push(currentWord);
            
            if (predictionBuffer.current.length > 7) {
              predictionBuffer.current.shift();
            }
            
            const counts = {};
            predictionBuffer.current.forEach(word => { counts[word] = (counts[word] || 0) + 1; });
            
            const mostFrequentWord = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
            
            if (counts[mostFrequentWord] >= 4) {
              setPrediction(mostFrequentWord.toUpperCase());
              setConfidence(Math.round(currentConf * 100));
            }
          } else {
             predictionBuffer.current = [];
          }
        }
      }
    };

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
        onFrame: async () => { 
            if (webcamRef.current && webcamRef.current.video) {
                await holistic.send({ image: webcamRef.current.video }); 
            }
        },
        width: 640, height: 480
      });
      camera.start();
    }
  }, [model]);

  return (
    <div style={{ width: "100vw", height: "100vh", backgroundColor: "#111", overflow: "hidden", position: "relative" }}>
      
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
        <Webcam 
          ref={webcamRef} 
          style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }} 
        />
        <canvas 
          ref={canvasRef} 
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", transform: "scaleX(-1)" }} 
        />

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