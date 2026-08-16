/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "gradient"],
  "DESCRIPTION": "A single very slow gradient front sweeping across the canvas with a soft noise edge. Built specifically for the whole-house bed, where anything with visible structure reads as strobing at fifty feet.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.2,  "MIN": -0.8, "MAX": 0.8 },
    { "NAME": "angle", "TYPE": "float", "DEFAULT": 0.25, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "width", "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.2,  "MAX": 1.0 },
    { "NAME": "grain", "TYPE": "float", "DEFAULT": 0.25, "MIN": 0.0,  "MAX": 0.8 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hg(vec2 p) { return fract(sin(dot(p, vec2(12.99, 78.23))) * 43758.5453); }
float ng(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hg(i), hg(i + vec2(1, 0)), f.x),
               mix(hg(i + vec2(0, 1)), hg(i + vec2(1, 1)), f.x), f.y);
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
    float a = angle * 3.14159;
    float proj = uv.x * cos(a) + uv.y * sin(a);
    // One soft front, wrapping. No repeating structure at all.
    float ph = fract(proj * 0.5 - t * 0.25);
    float v = smoothstep(0.0, width * 0.5, ph) * smoothstep(1.0, 1.0 - width * 0.5, ph);
    v = mix(v, v * (0.65 + 0.35 * ng(uv * 4.0 + t)), grain);
    v = clamp(v, 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
