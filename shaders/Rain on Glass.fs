/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Droplets clinging to a pane, swelling until they break and run, leaving thinning trails. A represented subject rather than a pattern -- people read it instantly as rain on a window.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.35, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "drops", "TYPE": "float", "DEFAULT": 9.0,  "MIN": 3.0,  "MAX": 22.0 },
    { "NAME": "trail", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.05, "MAX": 1.0 },
    { "NAME": "swell", "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hr(vec2 p){ return fract(sin(dot(p, vec2(31.7, 87.3))) * 43758.5453); }
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
    float acc = 0.0;
    for (int i = 0; i < 22; i++) {
        if (float(i) >= drops) break;
        float fi = float(i);
        float col = hr(vec2(fi, 1.0));
        float spd = 0.25 + 0.75 * hr(vec2(fi, 5.0));
        // Each drop hangs, then releases -- fract() gives the reset, and the
        // eased front half is the "clinging" part.
        float ph = fract(hr(vec2(fi, 9.0)) + t * spd);
        float hang = smoothstep(0.0, 0.35, ph);
        float y = 1.0 - mix(ph, ph * ph, swell);
        float dx = abs(uv.x - col);
        dx = min(dx, 1.0 - dx);
        float head = exp(-(dx * dx) * 900.0) * exp(-pow((uv.y - y) * 34.0, 2.0));
        // Trail lives ABOVE the head: where the drop came from.
        float above = max(0.0, uv.y - y);
        float tail = exp(-(dx * dx) * 1500.0) * exp(-above / max(trail * 0.3, 0.01)) * hang * 0.55;
        acc = max(acc, max(head, tail));
    }
    float v = clamp(acc, 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
