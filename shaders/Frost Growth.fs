/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Ice creeping across a windowpane: a ragged front advances from the edges, crystals branching ahead of it, then it thaws back. Slow and unmistakably seasonal.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.2, "MIN": -0.9, "MAX": 0.9 },
    { "NAME": "branch", "TYPE": "float", "DEFAULT": 0.7, "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "grain",  "TYPE": "float", "DEFAULT": 6.0, "MIN": 2.0,  "MAX": 16.0 },
    { "NAME": "melt",   "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hf(vec2 p){ return fract(sin(dot(p, vec2(57.1, 13.9))) * 43758.5453); }
float nf(vec2 p){
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(hf(i),hf(i+vec2(1,0)),f.x), mix(hf(i+vec2(0,1)),hf(i+vec2(1,1)),f.x), f.y);
}
float ff(vec2 p){
    float a=0.5, s=0.0;
    for (int i=0;i<4;i++){ s+=a*nf(p); p=p*2.13+vec2(2.7,1.1); a*=0.5; }
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
    // Distance from the nearest edge -- frost starts at the frame.
    float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) * 2.0;
    // The front breathes in and out: freeze, hold, thaw.
    float cycle = 0.5 - 0.5 * cos(t * 0.55);
    float front = mix(0.02, 1.05, mix(cycle, cycle * cycle, melt));
    // Crystal structure: ridged noise gives branching filaments rather than
    // blobs. Sharpening with abs() is what makes the fingers.
    float n = ff(uv * grain + vec2(0.0, t * 0.05));
    float ridge = 1.0 - abs(n * 2.0 - 1.0);
    ridge = pow(clamp(ridge, 0.0, 1.0), 2.2);
    // Crystals reach AHEAD of the front, so the boundary is feathered.
    float reach = front + branch * ridge * 0.35;
    float frozen = 1.0 - smoothstep(reach - 0.12, reach + 0.02, edge);
    float v = frozen * (0.25 + 0.85 * ridge);
    float _pv = clamp(v,0.0,1.0);
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
