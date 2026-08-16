/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Thin-film interference on a soap bubble or a puddle: broad smooth bands of value cycling through the full range as the film thickness drifts. Written as a clean value RAMP so a multi-colour palette reads as bands of colour moving, not one colour breathing.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "bands",  "TYPE": "float", "DEFAULT": 3.0, "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "warp",   "TYPE": "float", "DEFAULT": 0.8, "MIN": 0.0,  "MAX": 2.0 },
    { "NAME": "flow",   "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float ho(vec2 p){ return fract(sin(dot(p, vec2(19.7, 63.1))) * 43758.5453); }
float no(vec2 p){
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(ho(i),ho(i+vec2(1,0)),f.x), mix(ho(i+vec2(0,1)),ho(i+vec2(1,1)),f.x), f.y);
}
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
    // Film thickness: a slow, lumpy field that drifts and stretches.
    float th = no(uv * 1.7 + vec2(t * 0.2, t * 0.13));
    th += 0.5 * no(uv * 3.4 - vec2(t * 0.31, 0.0));
    th += warp * 0.35 * no(uv * 0.9 + vec2(0.0, -t * 0.17));
    // Thickness -> interference order. A triangle wave keeps the ramp
    // continuous in BOTH directions so bands slide rather than snapping back.
    float ord = th * bands + t * flow * 1.2;
    float v = abs(fract(ord) * 2.0 - 1.0);
    // Ease the extremes slightly so the brightest and darkest bands have body.
    v = v * v * (3.0 - 2.0 * v);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
