/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 rev3, HUGEL - Movin' To The Sun)",
  "CATEGORIES": ["2d", "generator", "light"],
  "DESCRIPTION": "Sunlight pouring down from above the canvas: broad rays of the first colour fanning from a sun just over the top edge, swinging slowly, over a warm sky that deepens toward the bottom, with dust motes glinting in the beams. Every matrix shows light from ABOVE, so ten of them read as one sky, not ten copies of a picture. `rate` is signed: the sway swings the rays back and forth on the beat.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.4,  "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "rays",   "TYPE": "float", "DEFAULT": 9.0,  "MIN": 3.0,  "MAX": 24.0 },
    { "NAME": "spread", "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.2,  "MAX": 1.0 },
    { "NAME": "motes",  "TYPE": "float", "DEFAULT": 0.08, "MIN": 0.0,  "MAX": 0.3 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 0.85, 0.4, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.45, 0.1, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [0.6, 0.1, 0.3, 1.0] }
  ]
}*/
// c1 = the rays, c2 = the sky near the sun, c3 = the sky low down.
float hs2(vec2 p) { return fract(sin(dot(p, vec2(63.7, 157.3))) * 43758.5453); }
void main() {
    vec2 uv = isf_FragNormCoord;
    // the sun sits above the top edge, swinging gently left and right
    vec2 sun = vec2(0.5 + 0.25 * sin(TIME * rate * 0.5), 1.35);
    vec2 d = uv - sun;
    float ang = atan(d.x, -d.y);                 // 0 = straight down
    float r = length(d);
    // rays: a sum of two angular waves so widths vary, slowly turning
    float a1 = sin(ang * rays + TIME * rate * 0.6);
    float a2 = sin(ang * rays * 0.53 - TIME * rate * 0.35 + 1.3);
    float ray = smoothstep(1.0 - spread, 1.0, 0.5 + 0.5 * (0.65 * a1 + 0.35 * a2));
    // rays fade with distance from the sun, but reach the bottom
    ray *= 0.55 + 0.45 * smoothstep(1.6, 0.4, r);
    // the sky: warm near the sun, deeper low down
    vec3 sky = mix(c3.rgb, c2.rgb, smoothstep(0.0, 1.0, uv.y)) * 0.55;
    vec3 col = mix(sky, c1.rgb, ray);
    // dust motes glinting inside the beams
    vec2 g = floor(uv * vec2(48.0, 24.0) + vec2(0.0, TIME * 0.3));
    float pick = step(1.0 - motes, hs2(g));
    float tw = pow(0.5 + 0.5 * sin(TIME * (2.0 + 3.0 * hs2(g + 2.2)) + 6.28 * hs2(g + 5.1)), 3.0);
    col = mix(col, vec3(1.0), pick * tw * ray * 0.85);
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
