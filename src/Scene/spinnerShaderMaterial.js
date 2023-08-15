import { extend } from "@react-three/fiber";
import { ShaderMaterial } from "three";

class SpinnerShaderMaterial extends ShaderMaterial {
  constructor() {
    super({
      vertexShader: `
        varying vec2 vUv;
        void main() {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xyz, 1.);
            vUv = uv;
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        uniform sampler2D imageTexture;


        void main() {
            vec4 texture = texture2D(imageTexture, vUv);
            gl_FragColor = vec4(0.5);
        }
      `,
      uniforms: {
        imageTexture: {
          value: null,
        },
      },
    });
  }

  get imageTexture() {
    return this.uniforms.imageTexture.value;
  }

  set imageTexture(value) {
    this.uniforms.imageTexture.value = value;
  }
}

extend({ SpinnerShaderMaterial });
