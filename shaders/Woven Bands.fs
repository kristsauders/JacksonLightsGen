/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "Horizontal and vertical bands crossing in a lattice, each axis breathing on its own phase. Rectilinear, which suits window matrices and roof lines where round patterns look imposed.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "cols",  "TYPE": "float", "DEFAULT": 6.0, "MIN": 2.0,  "MAX": 16.0 },
    { "NAME": "rows",  "TYPE": "float", "DEFAULT": 4.0, "MIN": 2.0,  "MAX": 16.0 },
    { "NAME": "weave", "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float band(float x, float n, float ph, float w) {
    float f = fract(x * n + ph);
    float d = abs(f - 0.5);
    return smoothstep(w, w * 0.3, d);
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
    float wv = mix(0.16, 0.42, 0.5 + 0.5 * sin(t * 0.9));
    float wh = mix(0.16, 0.42, 0.5 + 0.5 * cos(t * 0.7));
    float v1 = band(uv.x, cols, t * 0.15, wv);
    float v2 = band(uv.y, rows, -t * 0.11, wh);
    // Crossings read brightest, which is what makes it look woven.
    float v = max(v1, v2) * mix(1.0, 0.55 + 0.45 * v1 * v2 * 4.0, weave);
    v = clamp(v, 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
