import { MeshReflectorMaterial, useTexture } from "@react-three/drei";
import "./floorShaderMaterial";
import floorImg from "static/floor.jpg";
import floorHomeImg from "static/floor-home.jpg";
import * as THREE from "three";
const Floor = () => {
  const floorTexture = useTexture(floorImg);
  const floorHomeTexture = useTexture(floorHomeImg);

  const a = new THREE.Matrix4();
  a.set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
  return (
    <mesh
      position={[0, -0.02, 0]}
      receiveShadow
      rotation={[-Math.PI / 2, 0, 0]}
    >
      <planeGeometry args={[2000, 2000]} />
      <MeshReflectorMaterial
        resolution={4096}
        // mixBlur={0.4}
        mixStrength={1000}
        roughness={2}
        depthScale={1.2}
        minDepthThreshold={0.4}
        maxDepthThreshold={1.4}
        color="#050505"
        metalness={0.5}
      />
    </mesh>
    // <mesh
    //   position={[0, -0.02, 0]}
    //   receiveShadow
    //   rotation={[-Math.PI / 2, 0, 0]}
    // >
    //   <planeGeometry args={[2000, 2000]} />
    //   <floorShaderMaterial
    //     uRoughnessMapStrength={2.15}
    //     uMipStrength={1}
    //     uReflectionStrength={0.9}
    //     uRoughnessMapOffset={new THREE.Vector2(2.82, 6.18)}
    //     uRoughnessMapScale={1.06}
    //     tRoughnessMap={floorHomeTexture}
    //     tDiffuse={floorTexture}
    //     uRoughnessMapResolution={new THREE.Vector2(512, 512)}
    //     textureMatrix={a}
    //   />
    // </mesh>
  );
};

export default Floor;
