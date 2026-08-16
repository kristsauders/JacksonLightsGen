/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Slow topographic contour lines sliding over each other. Line-based rather than filled, so it stays legible on a matrix without lighting every pixel -- low coverage by design.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.25, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "levels", "TYPE": "float", "DEFAULT": 7.0,  "MIN": 3.0,  "MAX": 18.0 },
    { "NAME": "warp",   "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "weight", "TYPE": "float", "DEFAULT": 0.35, "MIN": 0.08, "MAX": 0.8 }
  ]
}*/
float hs(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float nz(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hs(i), hs(i + vec2(1, 0)), f.x),
               mix(hs(i + vec2(0, 1)), hs(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    // A drifting height field; contours are its level sets.
    float hgt = nz(uv * 2.5 + vec2(t * 0.4, -t * 0.25))
              + 0.5 * nz(uv * 5.0 - vec2(t * 0.2, t * 0.35)) * warp;
    float band = fract(hgt * levels);
    float d = min(band, 1.0 - band);              // distance to nearest contour
    float v = smoothstep(weight * 0.25, 0.0, d);
    gl_FragColor = vec4(vec3(v), v);
}
