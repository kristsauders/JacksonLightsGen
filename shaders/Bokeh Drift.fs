/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Soft out-of-focus discs at varying sizes drifting past each other, like lights seen through a rainy window. Gentle enough for quiet passages, with real depth from the size variation.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.25, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "count", "TYPE": "float", "DEFAULT": 10.0, "MIN": 3.0,  "MAX": 22.0 },
    { "NAME": "scale", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.15, "MAX": 1.0 },
    { "NAME": "soft",  "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.1,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float h1(float n) { return fract(sin(n * 78.233) * 43758.5453); }
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
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
