/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Sunlight through moving water onto a pool floor: bright wandering filaments that pool and part. Nothing in the library had this kind of concentrated, liquid brightness with mostly-dark space between.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.5, "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "scale", "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.3,  "MAX": 3.0 },
    { "NAME": "sharp", "TYPE": "float", "DEFAULT": 4.0, "MIN": 2.0,  "MAX": 10.0 },
    { "NAME": "warp",  "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.0, "MAX": 1.5 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
// c1 = the water (shown dim), c2/c3 = the light filaments.
//
// REWRITTEN 2026-10-10 to the canonical tileable-caustic formulation (inten
// 0.005, averaged before the 1.4 power). The first version divided by 0.15 and
// never averaged, so its accumulator sat around -5 and abs()^sharp saturated
// almost every pixel: measured at matrix resolution, alpha 0.96 everywhere --
// Krists: "the matrices are all just blue with barely anything different at
// the edges". This one measures ~half the canvas lit, filaments over dark water.
vec3 palcol(float t) {
    t = clamp(t, 0.0, 1.0);
    return mix(mix(c1.rgb, c2.rgb, smoothstep(0.0, 0.55, t)),
               c3.rgb, smoothstep(0.55, 1.0, t));
}
void main() {
    vec2 uv = isf_FragNormCoord;
    const float TAU = 6.28318;
    vec2 p = mod(uv * TAU * scale, TAU) - 250.0;
    float t = TIME * rate;
    vec2 i = p;
    float c = 1.0;
    const float inten = 0.005;
    for (int n = 0; n < 5; n++) {
        float tt = t * (1.0 - 3.5 / float(n + 1));
        i = p + warp * vec2(cos(tt - i.x) + sin(tt + i.y), sin(tt - i.y) + cos(tt + i.x));
        c += 1.0 / length(vec2(p.x / (sin(i.x + tt) / inten), p.y / (cos(i.y + tt) / inten)));
    }
    c /= 5.0;
    c = 1.17 - pow(c, 1.4);
    float v = clamp(pow(abs(c), sharp), 0.0, 1.0);
    vec3 water = c1.rgb * 0.3;
    vec3 col = mix(water, palcol(0.35 + 0.65 * v), smoothstep(0.08, 0.6, v));
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
