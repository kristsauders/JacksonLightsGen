/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Slow topographic contour lines sliding over each other. Line-based rather than filled, so it stays legible on a matrix without lighting every pixel -- low coverage by design.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.25, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "levels", "TYPE": "float", "DEFAULT": 7.0,  "MIN": 3.0,  "MAX": 18.0 },
    { "NAME": "warp",   "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "weight", "TYPE": "float", "DEFAULT": 0.35, "MIN": 0.08, "MAX": 0.8 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hs(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float nz(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hs(i), hs(i + vec2(1, 0)), f.x),
               mix(hs(i + vec2(0, 1)), hs(i + vec2(1, 1)), f.x), f.y);
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
    // A drifting height field; contours are its level sets.
    float hgt = nz(uv * 2.5 + vec2(t * 0.4, -t * 0.25))
              + 0.5 * nz(uv * 5.0 - vec2(t * 0.2, t * 0.35)) * warp;
    float band = fract(hgt * levels);
    float d = min(band, 1.0 - band);              // distance to nearest contour
    float v = smoothstep(weight * 0.25, 0.0, d);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
