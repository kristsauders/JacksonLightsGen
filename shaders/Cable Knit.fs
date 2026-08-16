/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Cable-knit sweater: braided ropes twisting over each other between panels of stocking stitch. Dense, tactile, and completely even -- the whole house becomes one knitted surface.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.2, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "cables", "TYPE": "float", "DEFAULT": 4.0, "MIN": 1.0,  "MAX": 10.0 },
    { "NAME": "twist",  "TYPE": "float", "DEFAULT": 3.0, "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "stitch", "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
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
    float cw = 1.0 / cables;
    float band = floor(uv.x / cw);
    float fx = fract(uv.x / cw);
    float phase = uv.y * twist + t + band * 1.7;
    // Two strands crossing: each is a sine in x, offset by half a period.
    float s1 = 0.5 + 0.35 * sin(phase);
    float s2 = 0.5 - 0.35 * sin(phase);
    float d1 = abs(fx - s1), d2 = abs(fx - s2);
    float rope = max(smoothstep(0.16, 0.02, d1), smoothstep(0.16, 0.02, d2));
    // Whichever strand is in front gets the highlight -- that is what makes
    // the braid read as over-and-under rather than as two flat waves.
    float front = step(d1, d2) * (0.5 + 0.5 * sin(phase));
    rope *= 0.6 + 0.4 * front;
    // Purl-stitch ground between the cables.
    float ground = (0.5 + 0.5 * sin(uv.x * cables * 18.0)) *
                   (0.5 + 0.5 * sin(uv.y * 90.0));
    float v = max(rope, ground * 0.35 * stitch);
    float _pv = clamp(v,0.0,1.0);
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
