/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "linear"],
  "DESCRIPTION": "Horizontal bands that swell and recede like a tide, with no left/right travel at all. Fills the gap left by removing the Wave effect without reintroducing a side-to-side scan.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "bands", "TYPE": "float", "DEFAULT": 5.0, "MIN": 2.0,  "MAX": 14.0 },
    { "NAME": "swell", "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "sharp", "TYPE": "float", "DEFAULT": 2.0, "MIN": 1.0,  "MAX": 6.0 }
  ]
}*/
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    // Each band breathes on its own phase; none of them translate sideways.
    float b = uv.y * bands;
    float idx = floor(b);
    float f = fract(b);
    float ph = sin(t * 1.1 + idx * 1.7);
    float half_w = mix(0.12, 0.5, 0.5 + 0.5 * ph * swell);
    float d = abs(f - 0.5);
    float v = smoothstep(half_w, half_w * 0.35, d);
    v = pow(v, sharp * 0.5);
    gl_FragColor = vec4(vec3(v), v);
}
