/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "linear"],
  "DESCRIPTION": "Horizontal bands that swell and recede like a tide, with no left/right travel at all. Fills the gap left by removing the Wave effect without reintroducing a side-to-side scan.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "bands", "TYPE": "float", "DEFAULT": 5.0, "MIN": 2.0,  "MAX": 14.0 },
    { "NAME": "swell", "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "sharp", "TYPE": "float", "DEFAULT": 2.0, "MIN": 1.0,  "MAX": 6.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
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
    // Each band breathes on its own phase; none of them translate sideways.
    float b = uv.y * bands;
    float idx = floor(b);
    float f = fract(b);
    float ph = sin(t * 1.1 + idx * 1.7);
    float half_w = mix(0.12, 0.5, 0.5 + 0.5 * ph * swell);
    float d = abs(f - 0.5);
    float v = smoothstep(half_w, half_w * 0.35, d);
    v = pow(v, sharp * 0.5);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
