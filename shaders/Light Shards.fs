/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 rev3, ODESZA - Light Of Day, the climax)",
  "CATEGORIES": ["2d", "generator", "light"],
  "DESCRIPTION": "Broad shafts of light sweeping diagonally across the canvas, each split like light through a prism -- a white-hot core with the palette colours fanning off its edges -- over a dark sky. Field-filling, so the whole house reads as one sky being cut by light. `rate` is signed: the beat-reversal sends the shafts back and forth on the beat.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.6,  "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "shafts", "TYPE": "float", "DEFAULT": 3.0,  "MIN": 1.0,  "MAX": 6.0 },
    { "NAME": "width",  "TYPE": "float", "DEFAULT": 0.30, "MIN": 0.05, "MAX": 0.4 },
    { "NAME": "spread", "TYPE": "float", "DEFAULT": 0.07, "MIN": 0.0,  "MAX": 0.15 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 0.78, 0.34, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.62, 0.5, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [0.53, 0.81, 0.92, 1.0] }
  ]
}*/
// c1, c2, c3 = the prism fan (c1 innermost); white is the core.
float shaft(float d, float w) { return pow(smoothstep(w, 0.0, abs(d)), 0.7); }
void main() {
    vec2 uv = isf_FragNormCoord;
    // diagonal coordinate across the shafts, swept over time
    float a = (uv.x * 0.8 + uv.y * 0.6) * shafts - TIME * rate * 0.8;
    float f = fract(a) - 0.5;
    // a second, slower, crossing set so the light braids
    float b = (uv.x * 0.6 - uv.y * 0.8) * shafts * 0.6 + TIME * rate * 0.35;
    float g = fract(b) - 0.5;
    vec3 col = vec3(0.0);
    // prism: each colour is the same shaft, offset outward
    col += c3.rgb * shaft(f + 2.0 * spread, width);
    col += c2.rgb * shaft(f + spread, width);
    col += c1.rgb * shaft(f, width);
    col += vec3(1.0) * shaft(f, width * 0.35);
    col += 0.55 * (c1.rgb * shaft(g, width * 0.8) + vec3(0.6) * shaft(g, width * 0.25));
    // the sky between the shafts: a low glow of the outer colour
    // measured 0.35 on the matrices at 0.08 / narrow shafts -- the climax must
    // not dim after chorus 1 (0.52): wider shafts, a brighter sky
    col += c3.rgb * 0.2;
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
