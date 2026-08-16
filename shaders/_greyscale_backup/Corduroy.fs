/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Ribbed corduroy with a sheen that travels along the ribs. Directional but even -- every part of the canvas looks like every other part, which is what lets it run house-wide.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.35, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "ribs",  "TYPE": "float", "DEFAULT": 9.0,  "MIN": 3.0,  "MAX": 28.0 },
    { "NAME": "sheen", "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "depth", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.1,  "MAX": 1.0 }
  ]
}*/
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    float r = fract(uv.x * ribs);
    // Rounded rib cross-section: bright along the crown, dark in the valley.
    float crown = pow(sin(r * 3.14159), mix(1.0, 4.0, depth));
    // A highlight sliding along the length of every rib together.
    float travel = 0.5 + 0.5 * sin(uv.y * 3.0 - t * 2.0 + floor(uv.x * ribs) * 0.4);
    float v = crown * mix(1.0, travel, sheen);
    gl_FragColor = vec4(vec3(clamp(v, 0.0, 1.0)), clamp(v, 0.0, 1.0));
}
