/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Flowing molten metal with rolling highlight bands. Continuous and centreless, with a sense of mass -- reads as a material, not a shape.",
  "INPUTS": [
    { "NAME": "rate", "TYPE": "float", "DEFAULT": 0.25, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "flow", "TYPE": "float", "DEFAULT": 3.0,  "MIN": 1.0,  "MAX": 9.0 },
    { "NAME": "bands","TYPE": "float", "DEFAULT": 5.0,  "MIN": 2.0,  "MAX": 14.0 },
    { "NAME": "gloss","TYPE": "float", "DEFAULT": 0.7,  "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
float hl(vec2 p){ return fract(sin(dot(p, vec2(12.99, 78.23))) * 43758.5453); }
float nl(vec2 p){
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(hl(i),hl(i+vec2(1,0)),f.x), mix(hl(i+vec2(0,1)),hl(i+vec2(1,1)),f.x), f.y);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    // Domain-warp the space, then read level bands off it: the classic
    // look of light rolling over a liquid metal surface.
    vec2 q = uv * flow;
    vec2 w = vec2(nl(q + t * 0.4), nl(q.yx - t * 0.3));
    float h = nl(q + w * 1.6);
    float v = 0.5 + 0.5 * sin(h * bands * 3.14159 + t * 1.2);
    // Sharpen the bright rolls so they read as specular, not as gradient.
    v = pow(v, mix(1.0, 5.0, gloss));
    gl_FragColor = vec4(vec3(clamp(v,0.0,1.0)), clamp(v,0.0,1.0));
}
