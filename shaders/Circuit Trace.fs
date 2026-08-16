/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "Angular circuit-board traces with pulses running along them. Rectilinear and evenly distributed, so it suits window matrices and rooflines and never collapses to a focal point.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.4, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "grid",  "TYPE": "float", "DEFAULT": 7.0, "MIN": 3.0,  "MAX": 20.0 },
    { "NAME": "trace", "TYPE": "float", "DEFAULT": 0.3, "MIN": 0.08, "MAX": 0.6 },
    { "NAME": "pulse", "TYPE": "float", "DEFAULT": 0.7, "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hc(vec2 p){ return fract(sin(dot(p, vec2(43.7, 191.3))) * 43758.5453); }
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
    vec2 g = uv * grid;
    vec2 cell = floor(g), f = fract(g) - 0.5;
    float r = hc(cell);
    // Each cell carries either a horizontal or a vertical trace, sometimes an
    // elbow -- enough variety to look routed rather than ruled.
    float line;
    if (r < 0.4)       line = smoothstep(trace * 0.5, 0.0, abs(f.y));
    else if (r < 0.8)  line = smoothstep(trace * 0.5, 0.0, abs(f.x));
    else               line = max(smoothstep(trace * 0.5, 0.0, abs(f.y)) * step(f.x, 0.0),
                                  smoothstep(trace * 0.5, 0.0, abs(f.x)) * step(0.0, f.y));
    // A charge travelling along the routing.
    float travel = 0.5 + 0.5 * sin((cell.x + cell.y) * 1.1 - t * 3.0);
    float v = line * mix(1.0, travel, pulse);
    // Solder pads where traces meet.
    float pad = smoothstep(0.16, 0.10, length(f)) * step(0.75, r);
    v = max(v, pad);
    float _pv = clamp(v,0.0,1.0);
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
