/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Crushed velvet: fine fibres lying in swirled directions, catching the light only where they face the sweep. A hand-brushed sheen crosses the canvas, lifting whole regions at once.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.25, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "swirl", "TYPE": "float", "DEFAULT": 0.7,  "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "fibre", "TYPE": "float", "DEFAULT": 60.0, "MIN": 15.0, "MAX": 160.0 },
    { "NAME": "sheen", "TYPE": "float", "DEFAULT": 0.7,  "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hn(vec2 p){ return fract(sin(dot(p, vec2(23.7, 91.3))) * 43758.5453); }
float nn(vec2 p){
    vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    return mix(mix(hn(i),hn(i+vec2(1,0)),f.x), mix(hn(i+vec2(0,1)),hn(i+vec2(1,1)),f.x), f.y);
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
    float _pv = clamp(v,0.0,1.0);
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
