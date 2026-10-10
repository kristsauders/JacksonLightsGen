/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 rev3, Harry Styles - Aperture)",
  "CATEGORIES": ["2d", "generator", "rhythmic"],
  "DESCRIPTION": "A honeycomb of camera irises: in every cell six blades turn to open and close the aperture, and the opening travels across the canvas in waves, light pouring through each open iris. Field-filling (no single centre), so ten matrices read as one wall of lenses. `rate` is signed: the beat-reversal on the speed curve makes the irises open and shut on the beat.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.5,  "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "scale",  "TYPE": "float", "DEFAULT": 4.0,  "MIN": 1.5,  "MAX": 10.0 },
    { "NAME": "wave",   "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 2.0 },
    { "NAME": "blades", "TYPE": "float", "DEFAULT": 0.3,  "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 0.96, 0.88, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [0.12, 0.56, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [0.75, 0.78, 0.82, 1.0] }
  ]
}*/
// c1 = the light through each iris, c2 = its rim glow, c3 = the blades (dim).
const float PI = 3.14159265;
vec4 hexCell(vec2 p) {
    vec2 r = vec2(1.0, 1.7320508);
    vec2 h = r * 0.5;
    vec2 a = mod(p, r) - h;
    vec2 b = mod(p - h, r) - h;
    vec2 gv = dot(a, a) < dot(b, b) ? a : b;
    return vec4(gv, p - gv);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    vec4 hc = hexCell(uv * vec2(scale, scale * 0.6));
    vec2 gv = hc.xy;
    vec2 id = hc.zw;
    // the opening travels as a wave across the field
    float ph = TIME * rate * 2.0 - (id.x * 0.9 + id.y * 0.5) * wave;
    float open = 0.08 + 0.34 * (0.5 + 0.5 * sin(ph));
    // six blades: the aperture is a hexagon that ROTATES as it opens
    float rot = open * 2.2;
    float ang = atan(gv.y, gv.x) + rot;
    float seg = 2.0 * PI / 6.0;
    float r = length(gv);
    float poly = r * cos(mod(ang, seg) - seg * 0.5) / cos(seg * 0.5);
    float inside = smoothstep(open + 0.02, open - 0.02, poly);
    // the blades: dim metal with curved seams sweeping in from the rim
    float seam = smoothstep(0.85, 1.0, abs(sin(6.0 * ang + r * 14.0)));
    vec3 blade = c3.rgb * (0.18 + 0.12 * (1.0 - seam)) * blades / 0.3;
    // light through the iris, brightest in the middle, rim in the second colour
    float core = 1.0 - smoothstep(0.0, open, poly) * 0.35;
    float rim = exp(-pow((poly - open) / 0.025, 2.0));
    vec3 col = mix(blade, c1.rgb * core, inside);
    col += c2.rgb * rim * 0.9;
    // cells fade to dark at their own edge so the honeycomb reads
    col *= smoothstep(0.5, 0.42, length(gv) * 0.95);
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
