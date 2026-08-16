/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Crushed velvet: fine fibres lying in swirled directions, catching the light only where they face the sweep. A hand-brushed sheen crosses the canvas, lifting whole regions at once.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.25, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "swirl", "TYPE": "float", "DEFAULT": 0.7,  "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "fibre", "TYPE": "float", "DEFAULT": 60.0, "MIN": 15.0, "MAX": 160.0 },
    { "NAME": "sheen", "TYPE": "float", "DEFAULT": 0.7,  "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
float hn(vec2 p){ return fract(sin(dot(p, vec2(23.7, 91.3))) * 43758.5453); }
float nn(vec2 p){
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(hn(i),hn(i+vec2(1,0)),f.x), mix(hn(i+vec2(0,1)),hn(i+vec2(1,1)),f.x), f.y);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    // Fibre lay direction: a slow swirling field, the crush pattern.
    float lay = (nn(uv * 2.3 + vec2(t * 0.15, -t * 0.1)) - 0.5) * 6.28318 * swirl;
    vec2 dir = vec2(cos(lay), sin(lay));
    // Fine fibres perpendicular to the lay.
    float fib = nn(vec2(dot(uv, vec2(-dir.y, dir.x)) * fibre, dot(uv, dir) * fibre * 0.18));
    // The sweep: a broad light source crossing, brightening fibres pointing
    // toward it. Same texture, two very different faces -- that is velvet.
    vec2 lightDir = vec2(cos(t * 0.8), sin(t * 0.8 * 0.6));
    float facing = 0.5 + 0.5 * dot(dir, normalize(lightDir));
    float v = mix(0.18, 1.0, facing) * mix(0.75, 1.0, fib);
    v *= mix(1.0, 0.55 + 0.75 * facing, sheen);
    gl_FragColor = vec4(vec3(clamp(v,0.0,1.0)), clamp(v,0.0,1.0));
}
