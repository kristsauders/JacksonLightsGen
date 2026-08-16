/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "radial"],
  "DESCRIPTION": "Two ripple sources whose wavefronts cross, producing moire bands that drift as the sources move. Complex-looking from a very simple rule, and the pattern is never the same twice.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.35, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "freq",  "TYPE": "float", "DEFAULT": 14.0, "MIN": 4.0,  "MAX": 40.0 },
    { "NAME": "sep",   "TYPE": "float", "DEFAULT": 0.3,  "MIN": 0.05, "MAX": 0.6 },
    { "NAME": "sharp", "TYPE": "float", "DEFAULT": 2.0,  "MIN": 1.0,  "MAX": 6.0 }
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
    vec2 a = vec2(0.5 - sep * 0.5 + 0.06 * sin(t * 0.7), 0.5 + 0.08 * cos(t * 0.5));
    vec2 b = vec2(0.5 + sep * 0.5 + 0.06 * cos(t * 0.6), 0.5 + 0.08 * sin(t * 0.8));
    float wa = sin(distance(uv, a) * freq - t * 3.0);
    float wb = sin(distance(uv, b) * freq - t * 3.0);
    float v = (wa + wb) * 0.5;
    v = pow(abs(v), sharp);          // bright where the crests agree
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
