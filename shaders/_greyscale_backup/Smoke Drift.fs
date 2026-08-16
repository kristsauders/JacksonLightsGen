/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Smoke curling upward, thinning as it rises. Domain-warped noise advected on a rising current -- soft, volumetric, and never repeating.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "curl",  "TYPE": "float", "DEFAULT": 0.8, "MIN": 0.0,  "MAX": 2.0 },
    { "NAME": "scale", "TYPE": "float", "DEFAULT": 3.0, "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "thin",  "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
float hk(vec2 p){ return fract(sin(dot(p, vec2(12.99, 78.23))) * 43758.5453); }
float nk(vec2 p){
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(hk(i),hk(i+vec2(1,0)),f.x), mix(hk(i+vec2(0,1)),hk(i+vec2(1,1)),f.x), f.y);
}
float fbk(vec2 p){
    float a=0.5, s=0.0;
    for (int i=0;i<5;i++){ s+=a*nk(p); p=p*2.07+vec2(1.3,-0.7); a*=0.5; }
    return s;
}
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    vec2 q = uv * scale;
    q.y -= t * 0.9;                            // the rising current
    // Curl the sampling position by the field itself -- what makes smoke
    // billow instead of merely scrolling.
    vec2 w = vec2(fbk(q + vec2(0.0, t * 0.3)), fbk(q.yx - vec2(t * 0.2, 0.0)));
    float dens = fbk(q + (w - 0.5) * curl * 2.0);
    // Thin out with height so it dissipates rather than filling the frame.
    dens *= mix(1.0, smoothstep(1.0, 0.05, uv.y), thin);
    float v = smoothstep(0.35, 0.75, dens);
    gl_FragColor = vec4(vec3(clamp(v,0.0,1.0)), clamp(v,0.0,1.0));
}
