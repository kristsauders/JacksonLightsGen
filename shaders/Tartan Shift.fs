/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Woven tartan: overlapping warp and weft bands, brightest where they cross. The colour of the sett flows smoothly from shade to shade while the weave itself stays still.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.4, "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "scale",   "TYPE": "float", "DEFAULT": 3.5, "MIN": 1.0,  "MAX": 10.0 },
    { "NAME": "stepdiv", "TYPE": "float", "DEFAULT": 2.0, "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "fine",    "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.0 }
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
float h21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
vec2 h22(vec2 p) {
    return fract(sin(vec2(dot(p, vec2(127.1, 311.7)),
                          dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}
float band(float x, float w) {
    float f = fract(x);
    return smoothstep(0.0, 0.09, f) * smoothstep(w, w - 0.09, f);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    float sx = uv.x * scale;
    float sy = uv.y * scale;
    // A wide stripe plus a narrow overcheck in each direction -- the overcheck
    // is what stops a plaid reading as a plain grid.
    float wx = band(sx, 0.58) + fine * band(sx * 2.0 + 0.31, 0.22);
    float wy = band(sy, 0.58) + fine * band(sy * 2.0 + 0.31, 0.22);
    float v = clamp(wx * 0.45 + wy * 0.45 + wx * wy * 0.55, 0.0, 1.0);
    // SMOOTH, not stepped (2026-10-10). The quantised march jumped on its own
    // clock, not the beat, and read as a glitch (Sleigh Ride 23:00, 23:35); the
    // colour now flows continuously and the beat-reversal on the speed curve
    // gives it its rhythm. `stepdiv` is kept for compatibility, unused.
    float t = TIME * rate * 0.5;
    float idx = floor(sx) * 0.31 + floor(sy) * 0.53 + t;
    gl_FragColor = vec4(palcol(0.5 + 0.5 * sin(6.2832 * idx * 0.37)), v);
}
