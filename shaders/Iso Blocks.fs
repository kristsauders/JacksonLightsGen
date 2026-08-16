/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "An isometric field of 3D blocks with shaded faces, rising and settling. Reads as dimensional depth rather than flat pattern, and tiles evenly so it can run across every canvas at once.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.28, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "tiles", "TYPE": "float", "DEFAULT": 5.0,  "MIN": 2.0,  "MAX": 14.0 },
    { "NAME": "rise",  "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "face",  "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.1,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hb(vec2 p){ return fract(sin(dot(p, vec2(21.98, 78.13))) * 43758.5453); }
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
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    // Skew into an isometric lattice: three rhombic faces per cube.
    vec2 sk = vec2(uv.x * 1.7320508 + uv.y, uv.y * 2.0) * tiles;
    vec2 cell = floor(sk);
    vec2 f = fract(sk);
    float h = hb(cell);
    // Each block bobs on its own phase; `rise` sets how much.
    float lift = 0.5 + 0.5 * sin(t * 1.3 + h * 6.28318);
    float top = step(f.x + f.y, 1.0);          // which half of the rhombus
    float left = step(f.x, f.y);
    // Three faces at three brightnesses -> reads as a lit solid.
    float shade = top * 1.0 + (1.0 - top) * mix(0.3, 0.55, left);
    shade = mix(shade, shade * (0.5 + 0.5 * lift), rise);
    float edge = smoothstep(0.0, 0.06, min(min(f.x, f.y), abs(f.x + f.y - 1.0)));
    float v = mix(1.0, shade, face) * mix(0.35, 1.0, edge);
    float _pv = clamp(v,0.0,1.0);
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
