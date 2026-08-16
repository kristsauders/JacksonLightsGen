/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "Angular crystalline shards that rotate slowly and catch light along their edges. Hard-edged where most of the vocabulary is soft, so it stands out as a deliberate change of texture.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.25, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "shards", "TYPE": "float", "DEFAULT": 5.0,  "MIN": 2.0,  "MAX": 14.0 },
    { "NAME": "facet",  "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.1,  "MAX": 1.0 },
    { "NAME": "glint",  "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hf(vec2 p) { return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453); }
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
    float best = 1e9, second = 1e9, id = 0.0;
    for (int i = 0; i < 14; i++) {
        if (float(i) >= shards) break;
        float fi = float(i);
        float ang = t * (0.3 + 0.4 * hf(vec2(fi, 1.0))) + hf(vec2(fi, 3.0)) * 6.28318;
        vec2 dir = vec2(cos(ang), sin(ang));
        // Signed distance to a rotating half-plane -- straight edges, not blobs.
        float d = abs(dot(uv, dir) - (hf(vec2(fi, 7.0)) - 0.5) * facet);
        if (d < best) { second = best; best = d; id = fi; }
        else if (d < second) { second = d; }
    }
    float body = smoothstep(0.28, 0.0, best);
    float edge = smoothstep(0.03, 0.0, abs(best - second)) * glint;
    float v = clamp(body * (0.35 + 0.35 * hf(vec2(id, 11.0))) + edge, 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
