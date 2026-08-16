/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Veined marble that drifts and re-forms. Turbulent, seamless and centreless -- looks like stone rather than like a pattern.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.22, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "veins", "TYPE": "float", "DEFAULT": 4.0,  "MIN": 1.0,  "MAX": 12.0 },
    { "NAME": "turb",  "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "sharp", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.1,  "MAX": 1.0 }
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
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    vec2 q = uv * 3.0 + vec2(t * 0.3, -t * 0.2);
    float warp = fbm(q + fbm(q * 1.7) * turb);
    // Veins are the ridges of the warped field -- abs() of a sine gives the
    // thin bright seams marble has.
    float v = 1.0 - abs(sin((uv.x + uv.y) * veins + warp * 6.0));
    v = pow(v, mix(6.0, 1.5, sharp));
    gl_FragColor = vec4(vec3(clamp(v,0.0,1.0)), clamp(v,0.0,1.0));
}
