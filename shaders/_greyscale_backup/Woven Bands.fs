/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "Horizontal and vertical bands crossing in a lattice, each axis breathing on its own phase. Rectilinear, which suits window matrices and roof lines where round patterns look imposed.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "cols",  "TYPE": "float", "DEFAULT": 6.0, "MIN": 2.0,  "MAX": 16.0 },
    { "NAME": "rows",  "TYPE": "float", "DEFAULT": 4.0, "MIN": 2.0,  "MAX": 16.0 },
    { "NAME": "weave", "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
float band(float x, float n, float ph, float w) {
    float f = fract(x * n + ph);
    float d = abs(f - 0.5);
    return smoothstep(w, w * 0.3, d);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    float wv = mix(0.16, 0.42, 0.5 + 0.5 * sin(t * 0.9));
    float wh = mix(0.16, 0.42, 0.5 + 0.5 * cos(t * 0.7));
    float v1 = band(uv.x, cols, t * 0.15, wv);
    float v2 = band(uv.y, rows, -t * 0.11, wh);
    // Crossings read brightest, which is what makes it look woven.
    float v = max(v1, v2) * mix(1.0, 0.55 + 0.45 * v1 * v2 * 4.0, weave);
    v = clamp(v, 0.0, 1.0);
    gl_FragColor = vec4(vec3(v), v);
}
