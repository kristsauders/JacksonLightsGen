/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "An isometric field of 3D blocks with shaded faces, rising and settling. Reads as dimensional depth rather than flat pattern, and tiles evenly so it can run across every canvas at once.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.28, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "tiles", "TYPE": "float", "DEFAULT": 5.0,  "MIN": 2.0,  "MAX": 14.0 },
    { "NAME": "rise",  "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "face",  "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.1,  "MAX": 1.0 }
  ]
}*/
float hb(vec2 p){ return fract(sin(dot(p, vec2(21.98, 78.13))) * 43758.5453); }
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    // Skew into an isometric lattice: three rhombic faces per cube.
    vec2 sk = vec2(uv.x * 1.7320508 + uv.y, uv.y * 2.0) * tiles;
    vec2 cell = floor(sk);
    vec2 f = fract(sk);
    float h = hb(cell);
    // Each block bobs on its own phase; `rise` sets how much.
    float lift = 0.5 + 0.5 * sin(t * 1.3 + h * 6.28318);
    float top = step(f.x + f.y, 1.0);          // which half of the rhombus
    float left = step(f.x, f.y);
    // Three faces at three brightnesses -> reads as a lit solid.
    float shade = top * 1.0 + (1.0 - top) * mix(0.3, 0.55, left);
    shade = mix(shade, shade * (0.5 + 0.5 * lift), rise);
    float edge = smoothstep(0.0, 0.06, min(min(f.x, f.y), abs(f.x + f.y - 1.0)));
    float v = mix(1.0, shade, face) * mix(0.35, 1.0, edge);
    gl_FragColor = vec4(vec3(clamp(v,0.0,1.0)), clamp(v,0.0,1.0));
}
