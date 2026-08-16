/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "particles"],
  "DESCRIPTION": "Sparse points drifting downward with a slow sway, leaving short trails. A snowfall/ember bed -- the vocabulary had no particle look, only washes and chases.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.35, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "density", "TYPE": "float", "DEFAULT": 14.0, "MIN": 4.0,  "MAX": 40.0 },
    { "NAME": "sway",    "TYPE": "float", "DEFAULT": 0.35, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "trail",   "TYPE": "float", "DEFAULT": 0.45, "MIN": 0.05, "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/

float hash(vec2 p) { return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453); }

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
    // Columns of falling points; each column has its own speed and phase.
    float col = floor(uv.x * density);
    float h1 = hash(vec2(col, 1.0));
    float h2 = hash(vec2(col, 7.0));
    float speed = 0.35 + 0.9 * h1;
    // Sway displaces the whole column horizontally, so flakes drift sideways.
    float dx = sway * 0.5 / density * sin(t * (0.6 + h2) + h1 * 6.28);
    float cx = (col + 0.5) / density + dx;
    float d = abs(uv.x - cx) * density;
    if (d > 1.0) { gl_FragColor = vec4(0.0); return; }
    float across = exp(-d * d * 6.0);
    // Vertical position of this column's particle, wrapping top to bottom.
    float y = fract(h2 + t * speed);
    float dy = uv.y - (1.0 - y);
    // Trail extends ABOVE the head (the direction it fell from).
    float along = dy > 0.0 ? exp(-dy / max(trail * 0.25, 0.01))
                           : exp(-(-dy) * 40.0);
    float v = clamp(across * along, 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
