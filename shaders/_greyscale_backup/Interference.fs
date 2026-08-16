/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "radial"],
  "DESCRIPTION": "Two ripple sources whose wavefronts cross, producing moire bands that drift as the sources move. Complex-looking from a very simple rule, and the pattern is never the same twice.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.35, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "freq",  "TYPE": "float", "DEFAULT": 14.0, "MIN": 4.0,  "MAX": 40.0 },
    { "NAME": "sep",   "TYPE": "float", "DEFAULT": 0.3,  "MIN": 0.05, "MAX": 0.6 },
    { "NAME": "sharp", "TYPE": "float", "DEFAULT": 2.0,  "MIN": 1.0,  "MAX": 6.0 }
  ]
}*/
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    vec2 a = vec2(0.5 - sep * 0.5 + 0.06 * sin(t * 0.7), 0.5 + 0.08 * cos(t * 0.5));
    vec2 b = vec2(0.5 + sep * 0.5 + 0.06 * cos(t * 0.6), 0.5 + 0.08 * sin(t * 0.8));
    float wa = sin(distance(uv, a) * freq - t * 3.0);
    float wb = sin(distance(uv, b) * freq - t * 3.0);
    float v = (wa + wb) * 0.5;
    v = pow(abs(v), sharp);          // bright where the crests agree
    gl_FragColor = vec4(vec3(v), v);
}
