/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "particles"],
  "DESCRIPTION": "A flock wheeling and condensing like starlings at dusk -- density gathers and disperses as the birds bank together. Movement with intent rather than a scrolling pattern.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "birds", "TYPE": "float", "DEFAULT": 60.0, "MIN": 15.0, "MAX": 140.0 },
    { "NAME": "bank",  "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "spread","TYPE": "float", "DEFAULT": 0.35, "MIN": 0.1, "MAX": 0.8 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hf(float n){ return fract(sin(n * 91.37) * 43758.5453); }
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
    // The flock's shared heading -- everything banks together.
    float head = t * 0.7;
    vec2 centre = vec2(0.5 + 0.22 * sin(head), 0.5 + 0.16 * cos(head * 1.3));
    // Breathing density: the flock condenses and opens out.
    float tight = mix(1.0, 0.35, bank * (0.5 + 0.5 * sin(t * 0.9)));
    float acc = 0.0;
    for (int i = 0; i < 140; i++) {
        if (float(i) >= birds) break;
        float fi = float(i);
        float a = hf(fi) * 6.28318 + head * (0.6 + 0.8 * hf(fi + 3.0));
        float r = spread * tight * (0.25 + 0.75 * hf(fi + 7.0));
        vec2 p = centre + vec2(cos(a), sin(a) * 0.6) * r;
        float d = distance(uv, p);
        acc += exp(-d * d * 2600.0);
    }
    float v = clamp(acc, 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
