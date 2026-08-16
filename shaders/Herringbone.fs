/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Woven herringbone twill. A textile texture with no focal point, so ten props can run it at once and it reads as one fabric rather than ten copies of a picture.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "scale", "TYPE": "float", "DEFAULT": 7.0, "MIN": 2.0,  "MAX": 20.0 },
    { "NAME": "slant", "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.3,  "MAX": 2.5 },
    { "NAME": "weft",  "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 }
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
    vec2 p = uv * scale;
    float band = floor(p.y);
    float dir = mod(band, 2.0) < 1.0 ? 1.0 : -1.0;      // alternate the twill
    float d = p.x * slant * dir + p.y + t;
    float w = fract(d);
    float v = smoothstep(0.0, 0.42, w) * smoothstep(1.0, 0.58, w);
    // Weft threads crossing the warp give it cloth grain.
    float cross_ = 0.5 + 0.5 * sin(p.y * 6.28318);
    v = mix(v, v * (0.55 + 0.45 * cross_), weft);
    float _pv = clamp(v, 0.0, 1.0);
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
