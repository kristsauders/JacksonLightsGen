/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Metaball blobs rising, merging and splitting under gentle convection. Very slow by design -- a whole-house piece for quiet passages where nothing should demand attention.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.18, "MIN": -0.8, "MAX": 0.8 },
    { "NAME": "blobs", "TYPE": "float", "DEFAULT": 6.0,  "MIN": 2.0,  "MAX": 12.0 },
    { "NAME": "mass",  "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.15, "MAX": 1.0 },
    { "NAME": "skin",  "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hv(float n){ return fract(sin(n * 57.31) * 43758.5453); }
// c1/c2/c3 are fed from the xLights palette (slots 1-3, in declaration
// order) -- see colorize_shaders.py for how that was established.
vec3 palcol(float t) {
    t = clamp(t, 0.0, 1.0);
    return mix(mix(c1.rgb, c2.rgb, smoothstep(0.0, 0.55, t)),
               c3.rgb, smoothstep(0.55, 1.0, t));
}
// Where in the palette this pixel sits: a slow diagonal drift, nudged by the
// shader's own value so bright detail still leans toward the palette's top
// end. Weighted toward the drift because value alone is invisible -- alpha
// hides exactly the part of the ramp that value would have shown.
float palpos(vec2 uv, float v) {
    float drift = 0.5 + 0.5 * sin((uv.x * 1.6 + uv.y * 1.1) * 2.1 - TIME * 0.11);
    return clamp(mix(drift, v, 0.35), 0.0, 1.0);
}

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
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
