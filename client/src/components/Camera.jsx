import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks, POSE_CONNECTIONS, HAND_CONNECTIONS } from "@mediapipe/drawing_utils";

const CLASS_LABELS = ['goodbye', 'go', 'hello', 'help', 'no', 'please', 'sorry', 'stop', 'thank you', 'yes'];

const CameraComponent = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState("Waiting...");
  const [confidence, setConfidence] = useState(0);

  // Buffer to store the last N predictions for smoothing
  const predictionBuffer = useRef([]); 

  useEffect(() => {
    const loadModel = async () => {
      try {
        const net = await tf.loadLayersModel("/model/model.json");
        setModel(net);
        setPrediction("Ready!");
      } catch (err) {
        setPrediction("Error");
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

      // Draw Logic
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
      const ctx = canvasRef.current.getContext("2d");
      ctx.save();
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Draw Skeleton (Optional: Remove if it's distracting)
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
      drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "#CC0000", lineWidth: 2 });
      drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "#00CC00", lineWidth: 2 });
      drawLandmarks(ctx, results.poseLandmarks, { color: "#FF0000", lineWidth: 1, radius: 3 }); 
      drawLandmarks(ctx, results.leftHandLandmarks, { color: "#00FF00", lineWidth: 1, radius: 3 }); 
      drawLandmarks(ctx, results.rightHandLandmarks, { color: "#FF0000", lineWidth: 1, radius: 3 }); 
      ctx.restore();

      // Inference Logic
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

          input.dispose();
          output.dispose();

          // --- SMOOTHING ALGORITHM ---
          // 1. Only accept if confidence is high (> 75%)
          if (currentConf > 0.75) {
            predictionBuffer.current.push(currentWord);
            
            // Keep buffer size at 8 frames
            if (predictionBuffer.current.length > 8) {
              predictionBuffer.current.shift();
            }

            // 2. CHECK CONSENSUS: Do the last 8 frames match?
            // This prevents "flickering" between similar words like Stop/Help
            const isStable = predictionBuffer.current.every(word => word === currentWord);

            if (isStable && predictionBuffer.current.length === 8) {
              setPrediction(currentWord.toUpperCase());
              setConfidence(Math.round(currentConf * 100));
            }
          }
        }
      }
    };

    const holistic = new Holistic({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}` });
    holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    holistic.onResults(onResults);

    if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null) {
      const camera = new Camera(webcamRef.current.video, {
        onFrame: async () => { await holistic.send({ image: webcamRef.current.video }); },
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, [model]);

  return (
    <div style={{ position: "relative", width: "640px", height: "480px" }}>
      {/* Overlay UI */}
      <div style={{
        position: "absolute",
        top: 20,
        left: 0,
        right: 0,
        textAlign: "center",
        zIndex: 20,
        color: "white",
        textShadow: "2px 2px 4px #000000",
      }}>
        <h2 style={{ fontSize: "50px", margin: 0, color: "#00FF00", textTransform: "uppercase" }}>
          {prediction}
        </h2>
        <p style={{ fontSize: "20px", margin: 0, fontWeight: "bold" }}>
          Accuracy: {confidence}%
        </p>
      </div>

      <Webcam ref={webcamRef} style={{ position: "absolute", left: 0, right: 0, zIndex: 9, width: 640, height: 480 }} />
      <canvas ref={canvasRef} style={{ position: "absolute", left: 0, right: 0, zIndex: 9, width: 640, height: 480 }} />
    </div>
  );
};

export default CameraComponent;