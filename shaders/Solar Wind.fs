/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 rev3, ODESZA - Light Of Day, the heavy last section of the climax)",
  "CATEGORIES": ["2d", "generator", "rhythmic"],
  "DESCRIPTION": "A storm of light rushing across the canvas: dense streaks of the palette colours at many speeds, bright heads and long fading tails, on lanes that ripple slightly. The whole stream reads as one force moving across the house. `rate` is signed: the beat-reversal flips the whole stream's direction on the beat.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.5,  "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "lanes",   "TYPE": "float", "DEFAULT": 4.0, "MIN": 4.0,  "MAX": 40.0 },
    { "NAME": "density", "TYPE": "float", "DEFAULT": 1.0,  "MIN": 0.1,  "MAX": 1.0 },
    { "NAME": "tail",    "TYPE": "float", "DEFAULT": 0.9, "MIN": 0.05, "MAX": 0.8 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 0.82, 0.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.24, 0.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [0.13, 0.31, 1.0, 1.0] }
  ]
}*/
// c1 = streak heads, c2 = tails, c3 = the dim lanes between.
float hw(float n) { return fract(sin(n * 127.1) * 43758.5453); }
void main() {
    vec2 uv = isf_FragNormCoord;
    // lanes ripple a little so the stream is not a ruled grid
    float y = uv.y * lanes + 0.35 * sin(uv.x * 6.0 + TIME * 0.7);
    float lane = floor(y);
    float ly = fract(y) - 0.5;
    float sp = 0.5 + 0.7 * hw(lane + 3.1);                // each lane its own speed (calmer: was 0.6-2.0)
    float x = uv.x * 2.0 - TIME * rate * sp;
    float seg = x * (1.5 + hw(lane + 7.7));
    float id = floor(seg);
    // the stream moves toward +x, so the head is the RIGHT end of a segment
    float p = 1.0 - fract(seg);                            // 0 = head ... 1 = end of tail
    float on = step(1.0 - density, hw(id * 1.3 + lane * 17.0));
    float head = smoothstep(0.0, 0.03, p) * smoothstep(tail, 0.0, p);
    float thick = smoothstep(0.5, 0.3, abs(ly));
    float v = on * head * thick;
    vec3 col = mix(c2.rgb, c1.rgb, smoothstep(tail * 0.6, 0.0, p)) * v;
    col += vec3(1.0) * on * thick * smoothstep(0.04, 0.0, p) * 0.7;       // white-hot heads
    // the lanes glow in the third colour, so the whole canvas carries colour
    // under the streaks (measured: at 0.12 the matrices sat at 0.11 mean)
    col += c3.rgb * 0.85 * smoothstep(0.5, 0.0, abs(ly));  // v10 0.20 at 0.28, v11 0.37 at 0.6
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
