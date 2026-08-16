/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "radial"],
  "DESCRIPTION": "Rings that fire from the centre and decay outward, each one a discrete event rather than a continuous scroll. Reads as a hit landing, so it layers well under a beat-locked brightness curve.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.9, "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "rings", "TYPE": "float", "DEFAULT": 4.0, "MIN": 1.0,  "MAX": 10.0 },
    { "NAME": "decay", "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.1,  "MAX": 1.0 },
    { "NAME": "width", "TYPE": "float", "DEFAULT": 0.3, "MIN": 0.05, "MAX": 0.8 }
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
    float r = length(uv) * 2.0;
    float acc = 0.0;
    for (int i = 0; i < 10; i++) {
        if (float(i) >= rings) break;
        // Each ring launches on its own offset and expands outward once.
        float age = fract(t * 0.5 + float(i) / max(rings, 1.0));
        float rr = age * 1.4;
        float d = abs(r - rr);
        float amp = pow(1.0 - age, 1.0 / max(decay, 0.05));
        // 0.02 made the rings thinner than the gaps between measured pixels, so
        // the expansion never registered on the house at all.
        acc += exp(-d * d / max(width * width * 0.08, 1e-6)) * amp;
    }
    float v = clamp(acc, 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
