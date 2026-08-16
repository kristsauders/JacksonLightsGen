/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Droplets clinging to a pane, swelling until they break and run, leaving thinning trails. A represented subject rather than a pattern -- people read it instantly as rain on a window.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.35, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "drops", "TYPE": "float", "DEFAULT": 9.0,  "MIN": 3.0,  "MAX": 22.0 },
    { "NAME": "trail", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.05, "MAX": 1.0 },
    { "NAME": "swell", "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
float hr(vec2 p){ return fract(sin(dot(p, vec2(31.7, 87.3))) * 43758.5453); }
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
    gl_FragColor = vec4(vec3(v), v);
}
