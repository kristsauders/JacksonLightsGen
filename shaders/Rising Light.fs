/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026, the Be With You build)",
  "CATEGORIES": ["2d", "generator", "fill"],
  "DESCRIPTION": "The house filling with light from the ground, once. Below `level`: two deep colours flowing slowly upward through each other, with breathing sparkles; at `level`: a soft glowing crest in the third colour; above: dark. `level` (0..1) is driven by a value curve so the rise happens exactly once over the build, wherever the clock is.",
  "INPUTS": [
    { "NAME": "level",   "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "crest",   "TYPE": "float", "DEFAULT": 0.05, "MIN": 0.01, "MAX": 0.2 },
    { "NAME": "flow",    "TYPE": "float", "DEFAULT": 0.35, "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "sparkle", "TYPE": "float", "DEFAULT": 0.06, "MIN": 0.0,  "MAX": 0.3 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [0.35, 0.05, 0.75, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [0.75, 0.02, 0.35, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 0.72, 0.2, 1.0] }
  ]
}*/
// c1, c2 = the two deep colours of the body; c3 = the crest.
float h2(vec2 p) { return fract(sin(dot(p, vec2(269.5, 183.3))) * 43758.5453); }
float vn(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h2(i), h2(i + vec2(1, 0)), f.x),
               mix(h2(i + vec2(0, 1)), h2(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int k = 0; k < 4; k++) { v += a * vn(p); p = p * 2.1 + vec2(3.1, 1.7); a *= 0.5; }
    return v;
}
void main() {
    vec2 uv = isf_FragNormCoord;
    // the crest is slightly wavy, like a surface, and moves with the flow
    float surf = level + 0.015 * sin(uv.x * 11.0 + TIME * 1.3) + 0.01 * sin(uv.x * 23.0 - TIME * 0.9);
    float d = surf - uv.y;                         // > 0 is below the surface
    float body = smoothstep(0.0, 0.03, d);
    // the body: two deep colours rising through each other
    vec2 q = vec2(uv.x * 3.0, uv.y * 3.0 - TIME * flow);
    float n = fbm(q + vec2(fbm(q + 1.7), fbm(q - 2.3)));
    vec3 col = mix(c1.rgb, c2.rgb, smoothstep(0.35, 0.7, n));
    col *= 0.55 + 0.45 * smoothstep(0.25, 0.75, n);
    // breathing sparkles inside the body
    vec2 g = floor(vec2(uv.x * 60.0, uv.y * 30.0 - TIME * flow * 30.0));
    float pick = step(1.0 - sparkle, h2(g));
    float br = 0.5 + 0.5 * sin(TIME * (2.0 + 3.0 * h2(g + 4.1)) + 6.28 * h2(g + 9.3));
    col = mix(col, vec3(1.0), pick * br * br * 0.8);
    // the crest: a glow on the surface in the third colour
    float cr = exp(-pow(d / crest, 2.0));
    vec3 outc = col * body + c3.rgb * cr;
    float a = clamp(max(body, cr), 0.0, 1.0);
    gl_FragColor = vec4(clamp(outc, 0.0, 1.0), a);
}
