/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Cable-knit sweater: braided ropes twisting over each other between panels of stocking stitch. Dense, tactile, and completely even -- the whole house becomes one knitted surface.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.2, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "cables", "TYPE": "float", "DEFAULT": 4.0, "MIN": 1.0,  "MAX": 10.0 },
    { "NAME": "twist",  "TYPE": "float", "DEFAULT": 3.0, "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "stitch", "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
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
    gl_FragColor = vec4(vec3(clamp(v,0.0,1.0)), clamp(v,0.0,1.0));
}
