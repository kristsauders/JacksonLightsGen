/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 warm-up)",
  "CATEGORIES": ["2d", "generator", "particles"],
  "DESCRIPTION": "A sparse field of single-pixel glints drifting slowly upward, each breathing in and out over seconds, with one soft comet crossing. Built for the warm-up teaser: full brightness where it is lit, but only a few percent of the canvas is ever lit, and nothing switches -- every glint fades in and out smoothly.",
  "INPUTS": [
    { "NAME": "density", "TYPE": "float", "DEFAULT": 0.06, "MIN": 0.01, "MAX": 0.2 },
    { "NAME": "cells",   "TYPE": "float", "DEFAULT": 18.0, "MIN": 6.0,  "MAX": 40.0 },
    { "NAME": "drift",   "TYPE": "float", "DEFAULT": 0.04, "MIN": 0.0,  "MAX": 0.2 },
    { "NAME": "comet",   "TYPE": "float", "DEFAULT": 1.0,  "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [0.8, 0.9, 1.0, 1.0] }
  ]
}*/
float h1(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void main() {
    vec2 uv = isf_FragNormCoord;
    vec2 g = vec2(uv.x * cells, uv.y * cells * 0.6 - TIME * drift * cells);
    vec2 id = floor(g), f = fract(g) - 0.5;
    float pick = h1(id);
    float v = 0.0;
    vec3 col = c1.rgb;
    if (pick < density) {
        vec2 o = vec2(h1(id + 3.1), h1(id + 7.7)) - 0.5;
        float dd = length(f - o * 0.6);
        // each glint breathes on its own slow cycle (0.15-0.4 Hz): a fade,
        // never a blink
        float br = 0.5 + 0.5 * sin(TIME * (0.9 + 1.6 * h1(id + 1.3)) + 6.28 * h1(id + 5.9));
        v = smoothstep(0.32, 0.0, dd) * br * br;
        col = mix(c1.rgb, c2.rgb, h1(id + 9.2));
    }
    // one comet, a soft streak crossing on a long diagonal
    float cx = fract(TIME * 0.05) * 1.6 - 0.3;
    float cy = 0.75 - 0.35 * (cx + 0.3);
    vec2 q = uv - vec2(cx, cy);
    float along = dot(q, normalize(vec2(-1.0, 0.45)));
    float across = abs(dot(q, normalize(vec2(0.45, 1.0))));
    float tail = smoothstep(0.0, 0.02, along) * smoothstep(0.3, 0.0, along);
    float head = smoothstep(0.025, 0.0, length(q));
    float cv = comet * max(head, tail * smoothstep(0.012, 0.0, across) * 0.7);
    v = max(v, cv);
    gl_FragColor = vec4(col * v, v);
}
