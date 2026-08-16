/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Metaball blobs rising, merging and splitting under gentle convection. Very slow by design -- a whole-house piece for quiet passages where nothing should demand attention.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.18, "MIN": -0.8, "MAX": 0.8 },
    { "NAME": "blobs", "TYPE": "float", "DEFAULT": 6.0,  "MIN": 2.0,  "MAX": 12.0 },
    { "NAME": "mass",  "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.15, "MAX": 1.0 },
    { "NAME": "skin",  "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
float hv(float n){ return fract(sin(n * 57.31) * 43758.5453); }
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    float field = 0.0;
    for (int i = 0; i < 12; i++) {
        if (float(i) >= blobs) break;
        float fi = float(i);
        float sp = 0.35 + 0.65 * hv(fi);
        // Rise, cool, sink: the vertical phase eases at both ends.
        float ph = fract(hv(fi + 4.0) + t * sp * 0.35);
        float y = 0.5 - 0.5 * cos(ph * 6.28318);
        float x = 0.5 + 0.34 * sin(t * sp * 0.8 + hv(fi + 8.0) * 6.28318);
        float r = (0.06 + 0.12 * hv(fi + 2.0)) * mass;
        vec2 d = uv - vec2(x, y);
        d.y *= 1.25;                          // slightly squashed, like fluid
        field += (r * r) / max(dot(d, d), 1e-5);
    }
    // Threshold the field: inside is bright, the boundary makes a taut skin.
    float v = smoothstep(0.85, 1.15, field);
    float edge = smoothstep(1.6, 1.0, field) * smoothstep(0.85, 1.05, field);
    v = clamp(v + edge * skin * 0.8, 0.0, 1.0);
    gl_FragColor = vec4(vec3(v), v);
}
