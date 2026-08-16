/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "particles"],
  "DESCRIPTION": "Sparse points drifting downward with a slow sway, leaving short trails. A snowfall/ember bed -- the vocabulary had no particle look, only washes and chases.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.35, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "density", "TYPE": "float", "DEFAULT": 14.0, "MIN": 4.0,  "MAX": 40.0 },
    { "NAME": "sway",    "TYPE": "float", "DEFAULT": 0.35, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "trail",   "TYPE": "float", "DEFAULT": 0.45, "MIN": 0.05, "MAX": 1.0 }
  ]
}*/

float hash(vec2 p) { return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453); }

void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    // Columns of falling points; each column has its own speed and phase.
    float col = floor(uv.x * density);
    float h1 = hash(vec2(col, 1.0));
    float h2 = hash(vec2(col, 7.0));
    float speed = 0.35 + 0.9 * h1;
    // Sway displaces the whole column horizontally, so flakes drift sideways.
    float dx = sway * 0.5 / density * sin(t * (0.6 + h2) + h1 * 6.28);
    float cx = (col + 0.5) / density + dx;
    float d = abs(uv.x - cx) * density;
    if (d > 1.0) { gl_FragColor = vec4(0.0); return; }
    float across = exp(-d * d * 6.0);
    // Vertical position of this column's particle, wrapping top to bottom.
    float y = fract(h2 + t * speed);
    float dy = uv.y - (1.0 - y);
    // Trail extends ABOVE the head (the direction it fell from).
    float along = dy > 0.0 ? exp(-dy / max(trail * 0.25, 0.01))
                           : exp(-(-dy) * 40.0);
    float v = clamp(across * along, 0.0, 1.0);
    gl_FragColor = vec4(vec3(v), v);
}
