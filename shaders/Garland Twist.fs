/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 rev2, Deck the Halls)",
  "CATEGORIES": ["2d", "generator", "festive"],
  "DESCRIPTION": "Twisted garland ribbons wound diagonally across the canvas, two colours braided over a lit ground of the third, with tinsel glints catching along the ribbon edges. Fills the whole canvas, for drops and choruses. Each ribbon's width breathes with the twist so it reads as a rope turning, not stripes scrolling. `rate` is signed: the generator's sway reverses the twist on the beat.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.6,  "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "count",   "TYPE": "float", "DEFAULT": 3.0,  "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "ground",  "TYPE": "float", "DEFAULT": 0.45, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "tinsel",  "TYPE": "float", "DEFAULT": 0.12, "MIN": 0.0,  "MAX": 0.4 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [0.04, 0.4, 0.14, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.84, 0.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [0.78, 0.06, 0.18, 1.0] }
  ]
}*/
// c1, c2 = the two braided ribbons; c3 = the lit ground between them.
float hg(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
    vec2 uv = isf_FragNormCoord;
    // diagonal coordinate across the ribbons, and along them
    float a = (uv.x + uv.y) * count;
    float along = (uv.x - uv.y) * 6.0;
    float ph = along + TIME * rate * 3.0;
    // two ropes per period, half a turn apart: each one's width swells and
    // thins with the twist, and they cross over where one is thin
    float f = fract(a);
    float w1 = 0.16 + 0.12 * cos(ph);
    float w2 = 0.16 - 0.12 * cos(ph);
    float d1 = abs(f - 0.25), d2 = abs(f - 0.75);
    float r1 = smoothstep(w1, w1 - 0.05, d1);
    float r2 = smoothstep(w2 + 0.0, w2 - 0.05, d2);
    // shading across each rope: brightest along its spine, so it looks round
    float s1 = 0.65 + 0.35 * (1.0 - d1 / max(w1, 0.02));
    float s2 = 0.65 + 0.35 * (1.0 - d2 / max(w2, 0.02));
    vec3 col = c3.rgb * ground;
    // the rope in front wins where they overlap: whichever is wider now
    if (w1 >= w2) { col = mix(col, c2.rgb * s2, r2); col = mix(col, c1.rgb * s1, r1); }
    else          { col = mix(col, c1.rgb * s1, r1); col = mix(col, c2.rgb * s2, r2); }
    // tinsel: glints along the rope edges, each twinkling on its own clock
    vec2 g = floor(vec2(along * 4.0, a * 8.0));
    float edge = max(smoothstep(0.06, 0.0, abs(d1 - w1)), smoothstep(0.06, 0.0, abs(d2 - w2)));
    float pick = step(1.0 - tinsel, hg(g));
    float tw = pow(0.5 + 0.5 * sin(TIME * (3.0 + 4.0 * hg(g + 3.3)) + 6.28 * hg(g + 7.7)), 3.0);
    col = mix(col, vec3(1.0), clamp(pick * tw * (0.4 + 0.6 * edge), 0.0, 1.0) * 0.9);
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
