/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Vertical aurora sheets that drift and fold. Slow, wide, never busy -- built for a whole-house bed or a big matrix where a fast pattern reads as strobing.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.35, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "sheets", "TYPE": "float", "DEFAULT": 3.0,  "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "fold",   "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.0,  "MAX": 1.4 },
    { "NAME": "lift",   "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/

float hash(float n) { return fract(sin(n) * 43758.5453123); }

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n = i.x + i.y * 57.0;
    return mix(mix(hash(n), hash(n + 1.0), f.x),
               mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y);
}

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
    float acc = 0.0;
    for (int i = 0; i < 8; i++) {
        if (float(i) >= sheets) break;
        float fi = float(i);
        // Each sheet is a soft vertical band whose x drifts on its own noise.
        float drift = noise(vec2(fi * 3.7, t * 0.6 + fi)) - 0.5;
        float cx = fract(fi / max(sheets, 1.0) + drift * fold + t * 0.07);
        float d = abs(uv.x - cx);
        d = min(d, 1.0 - d);                       // wrap around the canvas
        // Taller near the bottom, feathering out toward the top.
        float body = exp(-d * d * 220.0);
        float vert = mix(1.0, smoothstep(1.0, 0.15, uv.y), lift);
        float shimmer = 0.75 + 0.25 * noise(vec2(cx * 12.0, uv.y * 4.0 - t * 1.5));
        acc += body * vert * shimmer;
    }
    acc = clamp(acc, 0.0, 1.0);
    float _pv = acc;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
