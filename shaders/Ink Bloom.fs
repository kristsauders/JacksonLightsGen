/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "A drop of ink hitting water: it blooms outward with a ragged, noise-torn boundary, throws tendrils, then fades as the next drop lands. Centred by nature.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.35, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "tendril", "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "drops",   "TYPE": "float", "DEFAULT": 3.0,  "MIN": 1.0,  "MAX": 6.0 },
    { "NAME": "soft",    "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.05, "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hb(vec2 p){ return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453); }
float nb(vec2 p){
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(hb(i),hb(i+vec2(1,0)),f.x), mix(hb(i+vec2(0,1)),hb(i+vec2(1,1)),f.x), f.y);
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
    float v = 0.0;
    for (int i = 0; i < 6; i++) {
        if (float(i) >= drops) break;
        float fi = float(i);
        vec2 c = vec2(0.5) + 0.22 * vec2(sin(fi * 2.1), cos(fi * 3.7));
        vec2 d = uv - c;
        float r = length(d);
        float ang = atan(d.y, d.x);
        // Each drop has its own staggered lifecycle.
        float ph = fract(t * 0.35 + fi / max(drops, 1.0));
        float radius = 0.05 + 0.55 * sqrt(ph);       // fast bloom, slowing
        float fade = 1.0 - smoothstep(0.55, 1.0, ph);
        // The boundary is torn by angular noise that also creeps outward --
        // this is what makes it read as ink rather than a circle.
        float rag = nb(vec2(ang * 2.2, r * 5.0) + vec2(fi * 7.0, -ph * 1.5));
        float edge = radius * (1.0 + tendril * (rag - 0.5) * 0.9);
        v += fade * (1.0 - smoothstep(edge * (1.0 - soft * 0.45), edge, r));
    }
    v = clamp(v, 0.0, 1.0);
    float _pv = v;
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
