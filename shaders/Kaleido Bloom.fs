/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "kaleidoscope"],
  "DESCRIPTION": "Mirrored wedges blooming outward. Symmetric by construction, so it suits round props (flakes, spinners) where an asymmetric pattern looks accidental.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.4, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "wedges", "TYPE": "float", "DEFAULT": 6.0, "MIN": 3.0,  "MAX": 16.0 },
    { "NAME": "bloom",  "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "detail", "TYPE": "float", "DEFAULT": 3.0, "MIN": 1.0,  "MAX": 8.0 }
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
    vec2 uv = isf_FragNormCoord - 0.5;
    float t = TIME * rate;
    float a = atan(uv.y, uv.x);
    float r = length(uv) * 2.0;
    float seg = 6.28318 / wedges;
    a = abs(mod(a + t * 0.3, seg) - seg * 0.5);      // mirror into one wedge
    float v = 0.5 + 0.5 * sin(a * detail * 4.0 - r * 6.0 + t * 2.0);
    float envl = smoothstep(1.0, 0.15, r) * mix(1.0, 0.5 + 0.5 * sin(t * 1.3 - r * 3.0), bloom);
    v = clamp(pow(v, 2.0) * envl, 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
