/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Soft out-of-focus discs at varying sizes drifting past each other, like lights seen through a rainy window. Gentle enough for quiet passages, with real depth from the size variation.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.25, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "count", "TYPE": "float", "DEFAULT": 10.0, "MIN": 3.0,  "MAX": 22.0 },
    { "NAME": "scale", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.15, "MAX": 1.0 },
    { "NAME": "soft",  "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.1,  "MAX": 1.0 }
  ]
}*/
float h1(float n) { return fract(sin(n * 78.233) * 43758.5453); }
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    float acc = 0.0;
    for (int i = 0; i < 22; i++) {
        if (float(i) >= count) break;
        float fi = float(i);
        float depth = 0.35 + 0.65 * h1(fi + 5.0);      // near discs are bigger
        float rad = 0.05 + 0.16 * depth * scale;
        vec2 p = vec2(fract(h1(fi) + t * 0.09 * depth),
                      fract(h1(fi + 17.0) + t * 0.05 * depth));
        float d = distance(uv, p);
        // Bokeh: near-flat centre with a soft shoulder, not a gaussian point.
        float disc = smoothstep(rad, rad * (1.0 - 0.6 * soft), d);
        acc = max(acc, disc * (0.45 + 0.55 * depth));
    }
    float v = clamp(acc, 0.0, 1.0);
    gl_FragColor = vec4(vec3(v), v);
}
