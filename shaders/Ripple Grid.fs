/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "Concentric rings expanding from a moving origin over a faint grid. Reads as structure rather than texture -- the shader answer to a matrix that would otherwise just wash.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.6, "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "rings",   "TYPE": "float", "DEFAULT": 7.0, "MIN": 2.0,  "MAX": 20.0 },
    { "NAME": "thick",   "TYPE": "float", "DEFAULT": 0.4, "MIN": 0.08, "MAX": 0.9 },
    { "NAME": "wander",  "TYPE": "float", "DEFAULT": 0.3, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "gridMix", "TYPE": "float", "DEFAULT": 0.25, "MIN": 0.0, "MAX": 1.0 }
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
    vec2 o = vec2(0.5 + 0.35 * wander * sin(t * 0.41),
                  0.5 + 0.35 * wander * cos(t * 0.33));
    float r = distance(uv, o);
    float phase = fract(r * rings - t);
    // Sharp ring, soft shoulders -- thick controls how much of the period lights.
    float ring = smoothstep(0.0, thick * 0.5, phase) *
                 smoothstep(1.0, 1.0 - thick * 0.5, phase);
    ring = pow(ring, 2.0);
    // Faint lattice so the rings have something to move across.
    vec2 g = abs(fract(uv * 8.0) - 0.5);
    float grid = (1.0 - smoothstep(0.0, 0.06, min(g.x, g.y))) * gridMix;
    float v = clamp(ring + grid * (0.4 + 0.6 * ring), 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
