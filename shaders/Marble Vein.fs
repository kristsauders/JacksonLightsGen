/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Veined marble that drifts and re-forms. Turbulent, seamless and centreless -- looks like stone rather than like a pattern.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.22, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "veins", "TYPE": "float", "DEFAULT": 4.0,  "MIN": 1.0,  "MAX": 12.0 },
    { "NAME": "turb",  "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "sharp", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.1,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hm(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float nm(vec2 p){
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(hm(i),hm(i+vec2(1,0)),f.x), mix(hm(i+vec2(0,1)),hm(i+vec2(1,1)),f.x), f.y);
}
float fbm(vec2 p){
    float a=0.5, s=0.0;
    for (int i=0;i<4;i++){ s+=a*nm(p); p*=2.03; a*=0.5; }
    return s;
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
    vec2 q = uv * 3.0 + vec2(t * 0.3, -t * 0.2);
    float warp = fbm(q + fbm(q * 1.7) * turb);
    // Veins are the ridges of the warped field -- abs() of a sine gives the
    // thin bright seams marble has.
    float v = 1.0 - abs(sin((uv.x + uv.y) * veins + warp * 6.0));
    v = pow(v, mix(6.0, 1.5, sharp));
    float _pv = clamp(v,0.0,1.0);
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
