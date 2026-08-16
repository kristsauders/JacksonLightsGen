/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "Slow Voronoi cells breathing and sliding past each other. Organic and non-repeating, with no scan direction at all -- deliberately unlike every chase and sweep in the vocabulary.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "cells", "TYPE": "float", "DEFAULT": 4.0, "MIN": 1.5,  "MAX": 12.0 },
    { "NAME": "edge",  "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.05, "MAX": 1.0 },
    { "NAME": "pulse", "TYPE": "float", "DEFAULT": 0.3, "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    vec2 p = uv * cells;
    vec2 ip = floor(p), fp = fract(p);
    float d1 = 8.0, d2 = 8.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 h = hash2(ip + g);
            // Each seed orbits slowly on its own phase.
            vec2 o = g + 0.5 + 0.45 * sin(t + 6.2831 * h);
            float d = length(o - fp);
            if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
        }
    }
    // Edge distance: bright membranes between cells, dark interiors.
    float e = clamp((d2 - d1) / max(edge, 0.001), 0.0, 1.0);
    float v = 1.0 - e;
    v = pow(v, 1.6);
    // Gentle whole-field breath so it is never perfectly static.
    v *= 1.0 - pulse * 0.35 * (0.5 + 0.5 * sin(t * 1.7));
    v = clamp(v, 0.0, 1.0);
    gl_FragColor = vec4(vec3(v), v);
}
