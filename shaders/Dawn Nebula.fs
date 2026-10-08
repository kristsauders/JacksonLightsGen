/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 finale)",
  "CATEGORIES": ["2d", "generator", "space"],
  "DESCRIPTION": "Slow abstract nebula for the string-orchestra finale: layered, domain-warped clouds in three palette colours, drifting and folding into each other, with a few soft glowing knots. No lines, no horizon, no edges -- only texture that floats. Every term is a smooth function of position and TIME, so a shader speed of zero holds it perfectly still.",
  "INPUTS": [
    { "NAME": "scale",  "TYPE": "float", "DEFAULT": 2.2,  "MIN": 0.8, "MAX": 5.0 },
    { "NAME": "warp",   "TYPE": "float", "DEFAULT": 1.4,  "MIN": 0.0, "MAX": 3.0 },
    { "NAME": "drift",  "TYPE": "float", "DEFAULT": 0.06, "MIN": 0.0, "MAX": 0.3 },
    { "NAME": "glow",   "TYPE": "float", "DEFAULT": 0.35, "MIN": 0.0, "MAX": 1.0 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [0.12, 0.24, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.69, 0.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 0.16, 0.54, 1.0] }
  ]
}*/
// c1 = deep sky, c2 = gold, c3 = rose (palette slots 1-3).
float hs(vec2 p) { return fract(sin(dot(p, vec2(41.3, 289.1))) * 45758.5453); }
float vn(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hs(i), hs(i + vec2(1, 0)), f.x),
               mix(hs(i + vec2(0, 1)), hs(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int k = 0; k < 5; k++) { v += a * vn(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return v;
}
void main() {
    vec2 uv = isf_FragNormCoord * scale;
    float t = TIME * drift;
    // domain warp: the clouds fold through each other instead of sliding
    vec2 q = vec2(fbm(uv + vec2(0.0, t)), fbm(uv + vec2(5.2, -t * 0.7)));
    vec2 r = vec2(fbm(uv + warp * q + vec2(1.7, 9.2) + t * 0.5),
                  fbm(uv + warp * q + vec2(8.3, 2.8) - t * 0.4));
    float n = fbm(uv + warp * r);
    // three colours by two independent fields, so regions of each colour
    // interleave rather than banding by one value
    vec3 col = mix(c1.rgb, c3.rgb, smoothstep(0.35, 0.75, n));
    // gold tuned by measurement: 0.45-0.85 covered 10%, 0.38-0.62 over 55%
    col = mix(col, c2.rgb, smoothstep(0.45, 0.68, r.x) * 0.85);
    // soft glowing knots where the folds bunch up
    float k = smoothstep(0.62, 0.9, n * r.y * 1.6);
    col += glow * k * mix(c2.rgb, vec3(1.0), 0.4);
    // a little depth: darker in the thin parts of the cloud, never black
    col *= 0.55 + 0.45 * smoothstep(0.2, 0.7, n);
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
