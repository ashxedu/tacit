import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks, POSE_CONNECTIONS, HAND_CONNECTIONS } from "@mediapipe/drawing_utils";

// Sign Language Class Labels
const CLASS_LABELS = ['goodbye', 'go', 'hello', 'help', 'no', 'please', 'sorry', 'stop', 'thank you', 'yes'];

const CameraComponent = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState("Loading Model...");
  const [confidence, setConfidence] = useState(0);

  // 1. Load the Brain
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log("Loading model...");
        const net = await tf.loadLayersModel("/model/model.json");
        setModel(net);
        console.log("✅ Model Loaded!");
        setPrediction("Model Ready. Start Signing!");
      } catch (err) {
        console.error("❌ Failed to load model:", err);
        setPrediction("Error Loading Model");
      }
    };
    loadModel();
  }, []);

  // 2. The Data Formatter 
  const extractKeypoints = (results) => {
    // Pose: 33 landmarks * 4 values (x, y, z, visibility) = 132
    const pose = results.poseLandmarks 
      ? results.poseLandmarks.flatMap(res => [res.x, res.y, res.z, res.visibility])
      : new Array(33 * 4).fill(0);

    // Left Hand: 21 landmarks * 3 values (x, y, z) = 63
    const lh = results.leftHandLandmarks 
      ? results.leftHandLandmarks.flatMap(res => [res.x, res.y, res.z])
      : new Array(21 * 3).fill(0);

    // Right Hand: 21 landmarks * 3 values (x, y, z) = 63
    const rh = results.rightHandLandmarks 
      ? results.rightHandLandmarks.flatMap(res => [res.x, res.y, res.z])
      : new Array(21 * 3).fill(0);

    // Concatenate: [Pose, LH, RH] -> Total 258 values
    return [...pose, ...lh, ...rh];
  };

  // 3. The Prediction Loop
  useEffect(() => {
    // Prediction Buffer (Smoothing)
    let sequence = []; 

    const onResults = async (results) => {
      if (!canvasRef.current || !webcamRef.current || !webcamRef.current.video) return;

      // --- DRAWING (Visual Feedback) ---
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
      
      const ctx = canvasRef.current.getContext("2d");
      ctx.save();
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Draw Skeleton
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
      drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "#CC0000", lineWidth: 2 });
      drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "#00CC00", lineWidth: 2 });
      drawLandmarks(ctx, results.poseLandmarks, { color: "#FF0000", lineWidth: 1, radius: 3 }); 
      drawLandmarks(ctx, results.leftHandLandmarks, { color: "#00FF00", lineWidth: 1, radius: 3 }); 
      drawLandmarks(ctx, results.rightHandLandmarks, { color: "#FF0000", lineWidth: 1, radius: 3 }); 
      ctx.restore();

      // --- INFERENCE (The Brain) ---
      if (model) {
        const keypoints = extractKeypoints(results);
        sequence.push(keypoints);

        // Keep only last 30 frames (Matches Python training length)
        if (sequence.length > 30) sequence.shift();

        // Only predict if we have a full buffer of 30 frames
        if (sequence.length === 30) {
          // Wrap in tensor (Batch Size 1, 30 frames, 258 features)
          const input = tf.tensor([sequence]); 
          
          const output = model.predict(input);
          const values = await output.data();
          const maxIndex = values.indexOf(Math.max(...values));
          
          const predictedWord = CLASS_LABELS[maxIndex];
          const confidenceScore = values[maxIndex];

          // Threshold: Only show if > 60% confident
          if (confidenceScore > 0.7) {
            setPrediction(predictedWord.toUpperCase());
            setConfidence(Math.round(confidenceScore * 100));
          } else {
            setPrediction("..."); // Uncertainty
          }
          
          // Cleanup tensor memory (Critical for performance!)
          input.dispose();
          output.dispose();
        }
      }
    };

    const holistic = new Holistic({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
      },
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults(onResults);

    if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null) {
      const camera = new Camera(webcamRef.current.video, {
        onFrame: async () => {
          await holistic.send({ image: webcamRef.current.video });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, [model]);

  return (
    <div style={{ position: "relative", width: "640px", height: "480px" }}>
      {/* 1. Prediction Overlay */}
      <div style={{
        position: "absolute",
        top: 20,
        left: 0,
        right: 0,
        textAlign: "center",
        zIndex: 20,
        color: "white",
        textShadow: "2px 2px 4px #000000",
        fontFamily: "Arial, sans-serif"
      }}>
        <h2 style={{ fontSize: "40px", margin: 0, color: "#00FF00" }}>{prediction}</h2>
        <p style={{ fontSize: "18px", margin: 0 }}>Confidence: {confidence}%</p>
      </div>

      <Webcam
        ref={webcamRef}
        style={{ position: "absolute", left: 0, right: 0, zIndex: 9, width: 640, height: 480 }}
      />
      
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", left: 0, right: 0, zIndex: 9, width: 640, height: 480 }}
      />
    </div>
  );
};

export default CameraComponent;