/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "particles"],
  "DESCRIPTION": "A drifting swarm of soft points that circle a slowly moving centre. Sparse and alive without any scan direction -- good on big canvases during quiet passages.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "count",  "TYPE": "float", "DEFAULT": 9.0, "MIN": 3.0,  "MAX": 20.0 },
    { "NAME": "spread", "TYPE": "float", "DEFAULT": 0.32, "MIN": 0.05, "MAX": 0.5 },
    { "NAME": "size",   "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.15, "MAX": 1.0 }
  ]
}*/
float h(float n) { return fract(sin(n * 127.1) * 43758.5453); }
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    vec2 mid = vec2(0.5 + 0.12 * sin(t * 0.37), 0.5 + 0.12 * cos(t * 0.29));
    float acc = 0.0;
    for (int i = 0; i < 20; i++) {
        if (float(i) >= count) break;
        float fi = float(i);
        float sp = 0.4 + h(fi) * 1.2;
        float rad = spread * (0.35 + 0.65 * h(fi + 9.0));
        float ang = t * sp + h(fi + 3.0) * 6.28318;
        vec2 p = mid + vec2(cos(ang), sin(ang) * 0.7) * rad;
        float d = distance(uv, p);
        // 0.0006 put each ember at ~1% of the canvas across -- measured 0.01 lit,
        // i.e. invisible. This shader has been placed in songs and shown nothing.
        acc += exp(-d * d / max(0.006 * size * size, 1e-6));
    }
    float v = clamp(acc, 0.0, 1.0);
    gl_FragColor = vec4(vec3(v), v);
}
