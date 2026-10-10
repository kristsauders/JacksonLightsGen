/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 rev2, Bonobo - Shokoufeh)",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Shokoufeh is Persian for 'blossom': soft five-lobed petals drifting slowly across the canvas and turning as they go, in two palette colours, over a slow wash of the third. Smooth everywhere -- every petal fades in and out over seconds, nothing switches.",
  "INPUTS": [
    { "NAME": "density", "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.1, "MAX": 1.0 },
    { "NAME": "size",    "TYPE": "float", "DEFAULT": 0.42, "MIN": 0.15, "MAX": 0.8 },
    { "NAME": "drift",   "TYPE": "float", "DEFAULT": 0.12, "MIN": 0.0, "MAX": 0.5 },
    { "NAME": "wash",    "TYPE": "float", "DEFAULT": 0.35, "MIN": 0.0, "MAX": 1.0 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 0.5, 0.66, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.84, 0.9, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [0.18, 0.77, 0.71, 1.0] }
  ]
}*/
// c1, c2 = petal colours; c3 = the wash beneath.
float hb(vec2 p) { return fract(sin(dot(p, vec2(91.7, 413.3))) * 43758.5453); }
float vnb(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hb(i), hb(i + vec2(1, 0)), f.x), mix(hb(i + vec2(0, 1)), hb(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    // the wash: a slow, soft drift of the third colour
    float w = vnb(uv * 2.5 + vec2(TIME * 0.05, -TIME * 0.03));
    vec3 col = c3.rgb * wash * (0.4 + 0.6 * w);
    float a = wash * (0.4 + 0.6 * w);
    // petals on a drifting grid, one per cell where picked
    vec2 g = uv * 5.0 + vec2(TIME * drift, TIME * drift * 0.6);
    for (int ox = -1; ox <= 1; ox++) {
        for (int oy = -1; oy <= 1; oy++) {
            vec2 id = floor(g) + vec2(float(ox), float(oy));
            if (hb(id) > density) continue;
            vec2 ctr = id + 0.5 + 0.3 * (vec2(hb(id + 2.3), hb(id + 7.1)) - 0.5);
            vec2 d = g - ctr;
            float rot = TIME * (0.2 + 0.4 * hb(id + 3.7)) * (hb(id + 5.5) > 0.5 ? 1.0 : -1.0);
            float ang = atan(d.y, d.x) + rot;
            float r = length(d);
            float petal = size * (0.55 + 0.45 * pow(abs(cos(ang * 2.5)), 0.8));
            float m = smoothstep(petal, petal * 0.6, r);
            // each petal breathes in and out on its own slow cycle
            float life = 0.5 + 0.5 * sin(TIME * (0.25 + 0.3 * hb(id + 9.9)) + 6.28 * hb(id + 1.1));
            m *= smoothstep(0.1, 0.6, life);
            vec3 pc = mix(c1.rgb, c2.rgb, smoothstep(0.0, petal, r) * 0.8 + 0.2 * hb(id + 4.4));
            col = mix(col, pc, m);
            a = max(a, m);
        }
    }
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), clamp(a, 0.0, 1.0));
}
