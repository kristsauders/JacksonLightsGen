/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 rev3, BONNIE X CLYDE - What Love Can Do)",
  "CATEGORIES": ["2d", "generator", "rhythmic"],
  "DESCRIPTION": "A diamond lattice whose cells light in rings travelling outward, like a heartbeat spreading across the canvas: alternate rings in the first two colours, the lattice lines in the third. Geometric cells that glow up and fade as the rings pass, slowly enough never to strobe; `rate` is signed so the sway sends the rings outward and back on the beat, and `fill` sets how much of the lattice is lit at once.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.4,  "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "scale", "TYPE": "float", "DEFAULT": 6.0,  "MIN": 2.0,  "MAX": 16.0 },
    { "NAME": "fill",  "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.1,  "MAX": 1.0 },
    { "NAME": "lines", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 0.1, 0.35, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.75, 0.85, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [0.45, 0.0, 0.6, 1.0] }
  ]
}*/
// c1, c2 = alternate rings of lit cells; c3 = the lattice lines.
void main() {
    vec2 uv = isf_FragNormCoord - 0.5;
    // rotate 45 degrees: a square grid becomes diamonds
    vec2 p = vec2(uv.x + uv.y, uv.y - uv.x) * scale * 0.7071;
    vec2 id = floor(p);
    vec2 f = fract(p) - 0.5;
    // ring index of this cell: diamond (L1) distance from the centre
    float ring = abs(id.x + 0.5) + abs(id.y + 0.5);
    // Slow (v1 was 2.0x: "moving too fast, looks like it's strobing")
    float ph = ring * 0.35 - TIME * rate * 0.7;
    float w = fract(ph);
    // soft in and out -- a cell glows up and fades, it never snaps on/off
    float lit = smoothstep(0.0, 0.3, w) * smoothstep(fill, max(0.0, fill - 0.4), w);
    float which = mod(floor(ph), 2.0);
    vec3 cell = mix(c1.rgb, c2.rgb, which);
    // inset each cell so the lines show between them
    float inset = smoothstep(0.5, 0.42, max(abs(f.x), abs(f.y)));
    vec3 col = cell * lit * inset;
    // lattice lines in the third colour, always on
    float edge = 1.0 - inset;
    col = mix(col, c3.rgb, edge * lines);
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
