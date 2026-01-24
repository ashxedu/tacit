import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { POSE_CONNECTIONS, HAND_CONNECTIONS } from "@mediapipe/holistic";

const CameraComponent = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraReady, setCameraReady] = useState(false);

  useEffect(() => {
    const onResults = (results) => {
      if (!canvasRef.current || !webcamRef.current || !webcamRef.current.video) return;

      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set canvas dimensions to match video
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const ctx = canvasRef.current.getContext("2d");
      ctx.save();
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      // Draw Connectors (The Skeleton Lines)
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
      drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "#CC0000", lineWidth: 2 });
      drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "#00CC00", lineWidth: 2 });

      // Draw Landmarks (The Dots)
      drawLandmarks(ctx, results.poseLandmarks, { color: "#FF0000", lineWidth: 1, radius: 3 }); 
      drawLandmarks(ctx, results.leftHandLandmarks, { color: "#00FF00", lineWidth: 1, radius: 3 }); 
      drawLandmarks(ctx, results.rightHandLandmarks, { color: "#FF0000", lineWidth: 1, radius: 3 }); 

      ctx.restore();
    };

    // Initialize MediaPipe Holistic
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

    // Initialize Camera
    if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null) {
      const camera = new Camera(webcamRef.current.video, {
        onFrame: async () => {
          await holistic.send({ image: webcamRef.current.video });
        },
        width: 640,
        height: 480,
      });
      camera.start();
      setCameraReady(true);
    }
  }, []);

  return (
    <div style={{ position: "relative", width: "640px", height: "480px" }}>
      {/* 1. The Webcam Feed */}
      <Webcam
        ref={webcamRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zIndex: 9,
          width: 640,
          height: 480,
        }}
      />

      {/* 2. The Transparent Drawing Canvas */}
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zIndex: 9,
          width: 640,
          height: 480,
        }}
      />
    </div>
  );
};

export default CameraComponent;