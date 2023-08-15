import { OrbitControls } from "@react-three/drei";
import "./App.scss";
import { Canvas } from "@react-three/fiber";
import Scene from "Scene/Scene";
import Floor from "Floor/Floor";
import { Environment } from "@react-three/drei";
import { EffectComposer, Bloom, Noise } from "@react-three/postprocessing";
import * as THREE from "three";
const App = () => {
  return (
    <Canvas
      dpr={Math.max(window.devicePixelRatio, 2)}
      camera={{ position: [0, 0, 15] }}
      gl={{
        toneMappingExposure: 1.6,
        toneMapping: THREE.ACESFilmicToneMapping,
      }}
      onCreated={(state) => state.gl.setClearColor(0x050505)}
    >
      <Environment preset="city" />
      <OrbitControls />
      <Floor />
      <Scene />
      {/* <EffectComposer>
        <Bloom luminanceThreshold={0.0001} radius={0} intensity={2} />
      </EffectComposer> */}
    </Canvas>
  );
};

export default App;
