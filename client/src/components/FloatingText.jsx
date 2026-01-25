import React, { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Text3D, Center, Float } from "@react-three/drei";
import * as THREE from "three";

const FloatingText = ({ text }) => {
  const meshRef = useRef(null);

  // Animation: Gently rotate the text based on mouse/time
  useFrame((state) => {
    if (meshRef.current) {
      // Gentle bobbing motion
      meshRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.2;
      // Slight rotation
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
      <Center>
        <Text3D
          ref={meshRef}
          font="/fonts/helvetiker_regular.typeface.json"
          size={1.5}        // Size of letters
          height={0.5}      // Depth (Extrusion) of 3D letters
          curveSegments={12}
          bevelEnabled
          bevelThickness={0.05}
          bevelSize={0.02}
          bevelOffset={0}
          bevelSegments={5}
        >
          {text}
          <meshStandardMaterial
            color="#00FF00" // Matrix Green
            roughness={0.1}
            metalness={0.8}
            emissive="#004400"
          />
        </Text3D>
      </Center>
    </Float>
  );
};

export default FloatingText;