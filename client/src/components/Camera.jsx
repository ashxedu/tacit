import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks, POSE_CONNECTIONS, HAND_CONNECTIONS } from "@mediapipe/drawing_utils";

// ✅ UPDATED LIST (Alphabetical with 'intelligent')
const CLASS_LABELS = ['go', 'goodbye', 'hello', 'help', 'intelligent', 'no', 'please', 'sorry', 'stop', 'yes'];

const CameraComponent = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState("...");
  const [confidence, setConfidence] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [targetLabel, setTargetLabel] = useState("hello");

  // Buffer to store recent predictions
  const predictionBuffer = useRef([]); 

  useEffect(() => {
    const loadModel = async () => {
      try {
        const net = await tf.loadLayersModel("/model/model.json");
        setModel(net);
        setPrediction("System Ready");
      } catch (err) {
        setPrediction("Error Loading Model");
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

      // 1. Draw
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
      const ctx = canvasRef.current.getContext("2d");
      ctx.save();
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
      drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "#CC0000", lineWidth: 2 });
      drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "#00CC00", lineWidth: 2 });
      drawLandmarks(ctx, results.poseLandmarks, { color: "#FF0000", lineWidth: 1, radius: 3 }); 
      ctx.restore();

      // 2. Inference
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

          // --- STRICT SMOOTHING LOGIC ---
          
          // Rule 1: Confidence must be VERY high (> 85%)
          if (currentConf > 0.85) {
            predictionBuffer.current.push(currentWord);
            
            // Rule 2: Must match for 15 frames in a row (approx 0.5 seconds)
            if (predictionBuffer.current.length > 15) {
              predictionBuffer.current.shift();
            }

            const isStable = predictionBuffer.current.every(word => word === currentWord);
            
            if (isStable && predictionBuffer.current.length === 15) {
              setPrediction(currentWord.toUpperCase());
              setConfidence(Math.round(currentConf * 100));
            }
          } else {
            // Rule 3: If confidence drops, clear the buffer.
            // This prevents "stale" words from sticking around.
            predictionBuffer.current = [];
            // Optional: Uncomment next line if you want text to disappear when you stop signing
            // setPrediction("..."); 
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
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, [model]);

  // --- RECORDING FUNCTIONS ---
  const handleStartRecording = useCallback(() => {
    setIsRecording(true);
    setRecordedChunks([]);
    const stream = webcamRef.current.stream;
    mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: "video/webm" });
    mediaRecorderRef.current.addEventListener("dataavailable", ({ data }) => {
      if (data.size > 0) setRecordedChunks((prev) => [...prev, data]);
    });
    mediaRecorderRef.current.start();
  }, [webcamRef]);

  const handleStopRecording = useCallback(() => {
    mediaRecorderRef.current.stop();
    setIsRecording(false);
  }, [mediaRecorderRef]);

  useEffect(() => {
    if (!isRecording && recordedChunks.length > 0) {
      const blob = new Blob(recordedChunks, { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${targetLabel}_contribution_${Date.now()}.webm`; 
      a.click();
      window.URL.revokeObjectURL(url);
      setRecordedChunks([]);
    }
  }, [recordedChunks, isRecording, targetLabel]);

  return (
    <div style={{ position: "relative", width: "640px", height: "480px", margin: "auto" }}>
      
      {/* UI OVERLAY */}
      <div style={{ position: "absolute", top: 20, width: "100%", textAlign: "center", zIndex: 20, color: "white", textShadow: "2px 2px 4px #000" }}>
        <h2 style={{ fontSize: "50px", margin: 0, color: "#00FF00" }}>{prediction}</h2>
        <p style={{ fontSize: "18px", fontWeight: "bold", color: "#ddd"}}>
          {confidence > 0 ? `Confidence: ${confidence}%` : "Waiting for sign..."}
        </p>
      </div>

      <Webcam ref={webcamRef} style={{ position: "absolute", left: 0, top: 0, zIndex: 9, width: 640, height: 480 }} />
      <canvas ref={canvasRef} style={{ position: "absolute", left: 0, top: 0, zIndex: 10, width: 640, height: 480 }} />

      {/* CONTROLS */}
      <div style={{ position: "absolute", bottom: 20, width: "100%", textAlign: "center", zIndex: 30 }}>
        <select value={targetLabel} onChange={(e) => setTargetLabel(e.target.value)} style={{ padding: "10px", marginRight: "10px", borderRadius: "5px" }}>
          {CLASS_LABELS.map(word => <option key={word} value={word}>{word.toUpperCase()}</option>)}
        </select>
        {isRecording ? (
          <button onClick={handleStopRecording} style={{ padding: "10px 20px", backgroundColor: "red", color: "white", border: "none", borderRadius: "5px" }}>⏹ Stop</button>
        ) : (
          <button onClick={handleStartRecording} style={{ padding: "10px 20px", backgroundColor: "#007BFF", color: "white", border: "none", borderRadius: "5px" }}>🔴 Contribute Data</button>
        )}
      </div>
    </div>
  );
};

export default CameraComponent;