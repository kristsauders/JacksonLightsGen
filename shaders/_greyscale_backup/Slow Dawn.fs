/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "gradient"],
  "DESCRIPTION": "A single very slow gradient front sweeping across the canvas with a soft noise edge. Built specifically for the whole-house bed, where anything with visible structure reads as strobing at fifty feet.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.2,  "MIN": -0.8, "MAX": 0.8 },
    { "NAME": "angle", "TYPE": "float", "DEFAULT": 0.25, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "width", "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.2,  "MAX": 1.0 },
    { "NAME": "grain", "TYPE": "float", "DEFAULT": 0.25, "MIN": 0.0,  "MAX": 0.8 }
  ]
}*/
float hg(vec2 p) { return fract(sin(dot(p, vec2(12.99, 78.23))) * 43758.5453); }
float ng(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hg(i), hg(i + vec2(1, 0)), f.x),
               mix(hg(i + vec2(0, 1)), hg(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    float a = angle * 3.14159;
    float proj = uv.x * cos(a) + uv.y * sin(a);
    // One soft front, wrapping. No repeating structure at all.
    float ph = fract(proj * 0.5 - t * 0.25);
    float v = smoothstep(0.0, width * 0.5, ph) * smoothstep(1.0, 1.0 - width * 0.5, ph);
    v = mix(v, v * (0.65 + 0.35 * ng(uv * 4.0 + t)), grain);
    v = clamp(v, 0.0, 1.0);
    gl_FragColor = vec4(vec3(v), v);
}
