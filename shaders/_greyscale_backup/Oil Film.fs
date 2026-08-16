/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Thin-film interference on a soap bubble or a puddle: broad smooth bands of value cycling through the full range as the film thickness drifts. Written as a clean value RAMP so a multi-colour palette reads as bands of colour moving, not one colour breathing.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "bands",  "TYPE": "float", "DEFAULT": 3.0, "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "warp",   "TYPE": "float", "DEFAULT": 0.8, "MIN": 0.0,  "MAX": 2.0 },
    { "NAME": "flow",   "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
float ho(vec2 p){ return fract(sin(dot(p, vec2(19.7, 63.1))) * 43758.5453); }
float no(vec2 p){
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(ho(i),ho(i+vec2(1,0)),f.x), mix(ho(i+vec2(0,1)),ho(i+vec2(1,1)),f.x), f.y);
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
    gl_FragColor = vec4(vec3(v), v);
}
