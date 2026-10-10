/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 rev3, Elderbrook & George FitzGerald - Glad I Found You)",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Paper lanterns rising slowly into a dusk sky: soft glowing lanterns in two warm colours, each with a halo, drifting upward at their own pace and swaying a little, over a gradient of the third colour. Warm, hopeful, gentle; field-filling so it reads as one sky across the matrices. Smooth everywhere -- nothing switches.",
  "INPUTS": [
    { "NAME": "density", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.1, "MAX": 1.0 },
    { "NAME": "size",    "TYPE": "float", "DEFAULT": 0.22, "MIN": 0.08, "MAX": 0.45 },
    { "NAME": "rise",    "TYPE": "float", "DEFAULT": 0.12, "MIN": 0.0, "MAX": 0.5 },
    { "NAME": "sky",     "TYPE": "float", "DEFAULT": 0.4,  "MIN": 0.0, "MAX": 1.0 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 0.6, 0.2, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.85, 0.55, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [0.1, 0.55, 0.6, 1.0] }
  ]
}*/
// c1, c2 = lanterns; c3 = the sky.
float hl(vec2 p) { return fract(sin(dot(p, vec2(39.3, 271.9))) * 43758.5453); }
void main() {
    vec2 uv = isf_FragNormCoord;
    // the sky: the third colour, brighter low down where the town light is
    vec3 col = c3.rgb * sky * (0.45 + 0.55 * (1.0 - uv.y));
    vec2 g = vec2(uv.x * 6.0, uv.y * 4.0);
    for (int ox = -1; ox <= 1; ox++) {
        // each column rises at its own speed
        float colid = floor(g.x) + float(ox);
        float sp = rise * (0.6 + 0.8 * hl(vec2(colid, 3.3)));
        vec2 q = vec2(g.x, g.y - TIME * sp * 4.0);
        for (int oy = -1; oy <= 1; oy++) {
            vec2 id = vec2(colid, floor(q.y) + float(oy));
            if (hl(id) > density) continue;
            vec2 ctr = id + vec2(0.5 + 0.25 * sin(TIME * 0.5 + 6.28 * hl(id + 1.9)),
                                 0.5 + 0.2 * (hl(id + 4.7) - 0.5));
            vec2 d = q - ctr;
            d.y *= 0.8;                              // a little taller than wide
            float r = length(d);
            float s = size * (0.75 + 0.5 * hl(id + 8.8));
            float body = smoothstep(s, s * 0.55, r);
            float halo = exp(-pow(r / (s * 2.2), 2.0)) * 0.45;
            // the flame flickers softly, never off
            float fl = 0.8 + 0.2 * sin(TIME * (2.0 + 2.0 * hl(id + 6.1)) + 6.28 * hl(id + 2.4));
            vec3 lc = mix(c1.rgb, c2.rgb, hl(id + 5.5)) * fl;
            col = mix(col, lc, clamp(halo, 0.0, 1.0));
            col = mix(col, mix(lc, vec3(1.0), 0.25), body);
        }
    }
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
