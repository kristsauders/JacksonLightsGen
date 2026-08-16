/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "Concentric rings expanding from a moving origin over a faint grid. Reads as structure rather than texture -- the shader answer to a matrix that would otherwise just wash.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.6, "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "rings",   "TYPE": "float", "DEFAULT": 7.0, "MIN": 2.0,  "MAX": 20.0 },
    { "NAME": "thick",   "TYPE": "float", "DEFAULT": 0.4, "MIN": 0.08, "MAX": 0.9 },
    { "NAME": "wander",  "TYPE": "float", "DEFAULT": 0.3, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "gridMix", "TYPE": "float", "DEFAULT": 0.25, "MIN": 0.0, "MAX": 1.0 }
  ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    vec2 o = vec2(0.5 + 0.35 * wander * sin(t * 0.41),
                  0.5 + 0.35 * wander * cos(t * 0.33));
    float r = distance(uv, o);
    float phase = fract(r * rings - t);
    // Sharp ring, soft shoulders -- thick controls how much of the period lights.
    float ring = smoothstep(0.0, thick * 0.5, phase) *
                 smoothstep(1.0, 1.0 - thick * 0.5, phase);
    ring = pow(ring, 2.0);
    // Faint lattice so the rings have something to move across.
    vec2 g = abs(fract(uv * 8.0) - 0.5);
    float grid = (1.0 - smoothstep(0.0, 0.06, min(g.x, g.y))) * gridMix;
    float v = clamp(ring + grid * (0.4 + 0.6 * ring), 0.0, 1.0);
    gl_FragColor = vec4(vec3(v), v);
}
