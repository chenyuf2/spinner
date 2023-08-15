import { useGLTF, useTexture } from "@react-three/drei";
import spinner from "static/spinner-home.glb";
import env from "static/env.jpg";
import "./spinnerShaderMaterial";
import { useCallback, useRef } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";

const eulerRotation = new THREE.Euler(0, 0, 0.0024);
const rotationMatrix = new THREE.Matrix4();
rotationMatrix.makeRotationFromEuler(new THREE.Euler(0, 4.67, 0));

const RADIUS = 0.7;
const Scene = () => {
  const envTexture = useTexture(env);
  envTexture.needsUpdate = true;
  envTexture.colorSpace = THREE.SRGBColorSpace;
  const spinnerRef = useRef();
  const groupRef = useRef();
  const { gl } = useThree();
  const pmremGenerator = new THREE.PMREMGenerator(gl);
  pmremGenerator.compileEquirectangularShader();
  const { nodes } = useGLTF(spinner);
  const spinnerNodes = nodes.spinner;
  const beforeCompileFn = useCallback((shader) => {
    // e.uniforms = Object.assign(e.uniforms, this.userData.uniforms);
    shader.uniforms.uTime = { value: 0 };
    shader.uniforms.uThickness = { value: 0 };
    shader.uniforms.uEnvMapRotation = { value: rotationMatrix };
    shader.uniforms.uBlendColor = { value: new THREE.Color("#c1ff71") };
    shader.uniforms.uBlendColorStrength = { value: 0 };
    shader.vertexShader =
      `
    uniform float uThickness;

    ` + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace(
      "#include <uv_pars_vertex>",
      `
      varying vec2 vUv;
      `
    );

    shader.vertexShader = shader.vertexShader.replace(
      "#include <begin_vertex>",
      `
      vec3 pos = position;
      pos += uThickness * normal;
      vec3 transformed = vec3( pos );
      vUv = uv;
      `
    );

    shader.fragmentShader =
      `
      uniform mat4 uEnvMapRotation;
      // uniform vec3 uFunk;
      uniform float uTime;
      uniform vec3 uEmissiveColor;
      // uniform float uNormalNoiseScale;
      // uniform float uNormalNoiseOffsetSpeed;
      // uniform float uNormalNoiseStrength;
      // uniform float uNormalNoiseSpeed;
      uniform vec3 uBlendColor;      
      uniform float uBlendColorStrength;
      vec3 mod289(vec3 x) {
        return x - floor(x * (1.0 / 289.0)) * 289.0;
      }
      vec4 mod289(vec4 x) {
        return x - floor(x * (1.0 / 289.0)) * 289.0;
      }
      vec4 permute(vec4 x) {
        return mod289(((x*34.0)+1.0)*x);
      }
      vec4 taylorInvSqrt(vec4 r){
        return 1.79284291400159 - 0.85373472095314 * r;
      }

      float snoise(vec3 v){ 
        const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
        const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
      
      // First corner
        vec3 i  = floor(v + dot(v, C.yyy) );
        vec3 x0 =   v - i + dot(i, C.xxx) ;
      
      // Other corners
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min( g.xyz, l.zxy );
        vec3 i2 = max( g.xyz, l.zxy );
      
        //  x0 = x0 - 0. + 0.0 * C 
        vec3 x1 = x0 - i1 + 1.0 * C.xxx;
        vec3 x2 = x0 - i2 + 2.0 * C.xxx;
        vec3 x3 = x0 - 1. + 3.0 * C.xxx;
      
      // Permutations
        i = mod(i, 289.0 ); 
        vec4 p = permute( permute( permute( 
                  i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
                + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
      
      // Gradients
      // ( N*N points uniformly over a square, mapped onto an octahedron.)
        float n_ = 1.0/7.0; // N=7
        vec3  ns = n_ * D.wyz - D.xzx;
      
        vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)
      
        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
      
        vec4 x = x_ *ns.x + ns.yyyy;
        vec4 y = y_ *ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);
      
        vec4 b0 = vec4( x.xy, y.xy );
        vec4 b1 = vec4( x.zw, y.zw );
      
        vec4 s0 = floor(b0)*2.0 + 1.0;
        vec4 s1 = floor(b1)*2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));
      
        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
      
        vec3 p0 = vec3(a0.xy,h.x);
        vec3 p1 = vec3(a0.zw,h.y);
        vec3 p2 = vec3(a1.xy,h.z);
        vec3 p3 = vec3(a1.zw,h.w);
      
      //Normalise gradients
        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;
      
      // Mix final noise value
        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                      dot(p2,x2), dot(p3,x3) ) );
      }


      float blendOverlay(float base, float blend) {
        return base<0.5?(2.0*base*blend):(1.0-2.0*(1.0-base)*(1.0-blend));
      }
      vec3 blendOverlay(vec3 base, vec3 blend) {
          return vec3(blendOverlay(base.r,blend.r),blendOverlay(base.g,blend.g),blendOverlay(base.b,blend.b));
      }
      vec3 blendOverlay(vec3 base, vec3 blend, float opacity) {
          return (blendOverlay(base, blend) * opacity + base * (1.0 - opacity));
      }

      vec4 fromLinear(vec4 linearRGB) {
        bvec4 cutoff = lessThan(linearRGB, vec4(0.0031308));
        vec4 higher = vec4(1.055)*pow(linearRGB, vec4(1.0/2.4)) - vec4(0.055);
        vec4 lower = linearRGB * vec4(12.92);

        return mix(higher, lower, cutoff);
      }

      vec4 toLinear(vec4 sRGB) {
        bvec4 cutoff = lessThan(sRGB, vec4(0.04045));
        vec4 higher = pow((sRGB + vec4(0.055))/vec4(1.055), vec4(2.4));
        vec4 lower = sRGB/vec4(12.92);

        return mix(higher, lower, cutoff);
      }
  
  ` + shader.fragmentShader;

    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <envmap_physical_pars_fragment>",
      `
      #ifdef USE_ENVMAP

        vec3 getIBLIrradiance( const in vec3 normal ) {

          #ifdef ENVMAP_TYPE_CUBE_UV

            vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );

            vec4 envMapColor = textureCubeUV( envMap, worldNormal, 1.0 );

            return PI * envMapColor.rgb * envMapIntensity;

          #else

            return vec3( 0.0 );

          #endif

        }

        vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {

          #ifdef ENVMAP_TYPE_CUBE_UV

            vec3 reflectVec = reflect( - viewDir, normal );

            // Mixing the reflection with the normal is more accurate and keeps rough objects from gathering light from behind their tangent plane.
            reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );

            reflectVec = inverseTransformDirection( reflectVec, viewMatrix );

            
            //commented
            vec3 reflectVecTransformed = (uEnvMapRotation * vec4(reflectVec, 0.0)).xyz;

            vec4 envMapColor = textureCubeUV( envMap, reflectVecTransformed, roughness );

            return envMapColor.rgb * envMapIntensity;

          #else

            return vec3( 0.0 );

          #endif

        }

        #ifdef USE_ANISOTROPY

          vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {

            #ifdef ENVMAP_TYPE_CUBE_UV

              // https://google.github.io/filament/Filament.md.html#lighting/imagebasedlights/anisotropy
              vec3 bentNormal = cross( bitangent, viewDir );
              bentNormal = normalize( cross( bentNormal, bitangent ) );
              bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );

              return getIBLRadiance( viewDir, bentNormal, roughness );

            #else

              return vec3( 0.0 );

            #endif

          }

        #endif

      #endif

    `
    );

    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <cube_uv_reflection_fragment>",
      `
    #ifdef ENVMAP_TYPE_CUBE_UV
    
      #define cubeUV_minMipLevel 4.0
      #define cubeUV_minTileSize 16.0
    
      // These shader functions convert between the UV coordinates of a single face of
      // a cubemap, the 0-5 integer index of a cube face, and the direction vector for
      // sampling a textureCube (not generally normalized ).
    
      float getFace( vec3 direction ) {
    
        vec3 absDirection = abs( direction );
    
        float face = - 1.0;
    
        if ( absDirection.x > absDirection.z ) {
    
          if ( absDirection.x > absDirection.y )
    
            face = direction.x > 0.0 ? 0.0 : 3.0;
    
          else
    
            face = direction.y > 0.0 ? 1.0 : 4.0;
    
        } else {
    
          if ( absDirection.z > absDirection.y )
    
            face = direction.z > 0.0 ? 2.0 : 5.0;
    
          else
    
            face = direction.y > 0.0 ? 1.0 : 4.0;
    
        }
    
        return face;
    
      }
    
      // RH coordinate system; PMREM face-indexing convention
      vec2 getUV( vec3 direction, float face ) {
    
        vec2 uv;
    
        if ( face == 0.0 ) {
    
          uv = vec2( direction.z, direction.y ) / abs( direction.x ); // pos x
    
        } else if ( face == 1.0 ) {
    
          uv = vec2( - direction.x, - direction.z ) / abs( direction.y ); // pos y
    
        } else if ( face == 2.0 ) {
    
          uv = vec2( - direction.x, direction.y ) / abs( direction.z ); // pos z
    
        } else if ( face == 3.0 ) {
    
          uv = vec2( - direction.z, direction.y ) / abs( direction.x ); // neg x
    
        } else if ( face == 4.0 ) {
    
          uv = vec2( - direction.x, direction.z ) / abs( direction.y ); // neg y
    
        } else {
    
          uv = vec2( direction.x, direction.y ) / abs( direction.z ); // neg z
    
        }
    
        return 0.5 * ( uv + 1.0 );
    
      }
    
      vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
    
        float face = getFace( direction );
    
        float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
    
        mipInt = max( mipInt, cubeUV_minMipLevel );
    
        float faceSize = exp2( mipInt );
    
        highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0; // #25071
    
        if ( face > 2.0 ) {
    
          uv.y += faceSize;
    
          face -= 3.0;
    
        }
    
        uv.x += face * faceSize;
    
        uv.x += filterInt * 3.0 * cubeUV_minTileSize;
    
        uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
    
        uv.x *= CUBEUV_TEXEL_WIDTH;
        uv.y *= CUBEUV_TEXEL_HEIGHT;
    
        #ifdef texture2DGradEXT
    
          return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb; // disable anisotropic filtering
    
        #else
    
          return texture2D( envMap, uv ).rgb;
    
        #endif
    
      }
    
      // These defines must match with PMREMGenerator
    
      #define cubeUV_r0 1.0
      #define cubeUV_v0 0.339
      #define cubeUV_m0 - 2.0
      #define cubeUV_r1 0.8
      #define cubeUV_v1 0.276
      #define cubeUV_m1 - 1.0
      #define cubeUV_r4 0.4
      #define cubeUV_v4 0.046
      #define cubeUV_m4 2.0
      #define cubeUV_r5 0.305
      #define cubeUV_v5 0.016
      #define cubeUV_m5 3.0
      #define cubeUV_r6 0.21
      #define cubeUV_v6 0.0038
      #define cubeUV_m6 4.0
    
      float roughnessToMip( float roughness ) {
    
        float mip = 0.0;
    
        if ( roughness >= cubeUV_r1 ) {
    
          mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
    
        } else if ( roughness >= cubeUV_r4 ) {
    
          mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
    
        } else if ( roughness >= cubeUV_r5 ) {
    
          mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
    
        } else if ( roughness >= cubeUV_r6 ) {
    
          mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
    
        } else {
    
          mip = - 2.0 * log2( 1.16 * roughness ); // 1.16 = 1.79^0.25
        }
    
        return mip;
    
      }
    
      vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
    
        float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
    
        float mipF = fract( mip );
    
        float mipInt = floor( mip );
    
        vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
    
        if ( mipF == 0.0 ) {
    
          return vec4( color0, 1.0 );
    
        } else {
    
          vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
    
          return vec4( mix( color0, color1, mipF ), 1.0 );
    
        }
    
      }
    
    #endif
    `
    );

    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <uv_pars_fragment>",
      `
      varying vec2 vUv;
      
      `
    );

    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <normal_fragment_begin>",
      `
        float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;

        #ifdef FLAT_SHADED

          vec3 fdx = dFdx( vViewPosition );
          vec3 fdy = dFdy( vViewPosition );
          vec3 normal = normalize( cross( fdx, fdy ) );

        #else

          vec3 normal = normalize( vNormal );

          #ifdef DOUBLE_SIDED

            normal *= faceDirection;

          #endif

        #endif

        #if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )

          #ifdef USE_TANGENT

            mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );

          #else

            mat3 tbn = getTangentFrame( - vViewPosition, normal,
            #if defined( USE_NORMALMAP )
              vNormalMapUv
            #elif defined( USE_CLEARCOAT_NORMALMAP )
              vClearcoatNormalMapUv
            #else
              vUv
            #endif
            );

          #endif

          #if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )

            tbn[0] *= faceDirection;
            tbn[1] *= faceDirection;

          #endif

        #endif

        #ifdef USE_CLEARCOAT_NORMALMAP

          #ifdef USE_TANGENT

            mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );

          #else

            mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );

          #endif

          #if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )

            tbn2[0] *= faceDirection;
            tbn2[1] *= faceDirection;

          #endif

        #endif

        // non perturbed normal for clearcoat among others

        vec3 geometryNormal = normal;

      `
    );
    // console.log(shader.fragmentShader);
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <opaque_fragment>",
      `
      #ifdef OPAQUE
      diffuseColor.a = 1.0;
      #endif
      
      #ifdef USE_TRANSMISSION
      diffuseColor.a *= material.transmissionAlpha;
      #endif
      
      vec3 blendedColor = blendOverlay(outgoingLight, uBlendColor, uBlendColorStrength);
      gl_FragColor = vec4( blendedColor, diffuseColor.a );
      `
    );
    spinnerRef.current.material.userData.shader = shader;
  }, []);

  useFrame((state, delta) => {
    if (!spinnerRef.current) return;
    if (!groupRef.current) return;
    spinnerRef.current.rotation.y += 0.5 + Math.random() * 0.5;
    groupRef.current.position.x =
      RADIUS * Math.cos(state.clock.getElapsedTime());
    groupRef.current.position.z =
      RADIUS * Math.sin(state.clock.getElapsedTime());

    if (spinnerRef?.current?.material?.userData?.shader?.uniforms) {
      spinnerRef.current.material.userData.shader.uniforms.uTime.value += delta;
      spinnerRef.current.material.userData.shader.uniforms.uEnvMapRotation.value.makeRotationFromEuler(
        eulerRotation
      );
    }
  });
  return (
    <group ref={groupRef}>
      <mesh
        geometry={spinnerNodes.geometry}
        position={spinnerNodes.position}
        rotation={spinnerNodes.rotation}
        scale={spinnerNodes.scale}
        ref={spinnerRef}
      >
        <meshStandardMaterial
          metalness={1}
          roughness={0.29}
          envMap={pmremGenerator.fromEquirectangular(envTexture).texture}
          onBeforeCompile={beforeCompileFn}
          envMapIntensity={3}
          color={"#888888"}
        />
      </mesh>
    </group>
  );
};

export default Scene;
