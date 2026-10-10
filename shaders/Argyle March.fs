/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Knitted argyle lozenges that never move; only their colour flows, one diagonal row after the next. Rhythm without motion, so it is safe on the biggest props.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.5, "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "scale",   "TYPE": "float", "DEFAULT": 4.0, "MIN": 1.5,  "MAX": 12.0 },
    { "NAME": "stepdiv", "TYPE": "float", "DEFAULT": 2.0, "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "lineamt", "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.0 }
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
// Colour advances in DISCRETE STEPS rather than drifting: each diamond holds
// its colour for a beat's worth of time and then jumps to the next. That jump
// is the rhythmic element -- the shape never moves, only the colour marches,
// so it can run on a big prop without anything sweeping across it.
float diamond(vec2 p) {
    return abs(p.x) + abs(p.y);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    vec2 p = uv * scale;
    vec2 cell = floor(p);
    vec2 f = fract(p) - 0.5;
    float d = diamond(f) * 2.0;
    // Nested diamonds: a filled lozenge with a thin outline just outside it.
    float body = smoothstep(0.86, 0.72, d);
    float line = smoothstep(0.99, 0.93, d) * smoothstep(0.86, 0.90, d);
    float v = max(body * 0.85, line * lineamt);
    // The march: quantised time, plus a diagonal offset so the colour change
    // travels across the lattice one row at a time instead of all at once.
    // SMOOTH, not stepped (2026-10-10). The quantised march jumped on its own
    // clock, not the beat, and read as a glitch (Sleigh Ride 23:00, 23:35); the
    // colour now flows continuously and the beat-reversal on the speed curve
    // gives it its rhythm. `stepdiv` is kept for compatibility, unused.
    float t = TIME * rate * 0.5;
    float idx = t + (cell.x + cell.y) * 0.25;
    // a smooth CYCLE through the palette, never a wrap from c3 back to c1
    float hue = 0.5 + 0.5 * sin(6.2832 * idx * 0.37);
    float _pv = clamp(v, 0.0, 1.0);
    gl_FragColor = vec4(palcol(hue), _pv);
}
