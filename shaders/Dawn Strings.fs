/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 finale)",
  "CATEGORIES": ["2d", "generator", "gradient"],
  "DESCRIPTION": "A dawn horizon for the string-orchestra outro: gold light below, a royal-blue sky above, a rose glow on the horizon and faint gold rays. The horizon TILTS with the `bow` input (-1..1), which a value curve swings to alternate extremes on every bow change, so it sways right-left-right with the strings; holding the curve holds it perfectly still. No flicker anywhere: every term is a smooth function of position and of TIME alone.",
  "INPUTS": [
    { "NAME": "height", "TYPE": "float", "DEFAULT": 0.45, "MIN": 0.15, "MAX": 0.85 },
    { "NAME": "sway",   "TYPE": "float", "DEFAULT": 0.30, "MIN": 0.0,  "MAX": 0.6 },
    { "NAME": "soft",   "TYPE": "float", "DEFAULT": 0.10, "MIN": 0.02, "MAX": 0.35 },
    { "NAME": "rays",   "TYPE": "float", "DEFAULT": 0.22, "MIN": 0.0,  "MAX": 0.6 },
    { "NAME": "bow",    "TYPE": "float", "DEFAULT": 0.0,  "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [0.12, 0.24, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.69, 0.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 0.16, 0.54, 1.0] }
  ]
}*/
// c1 = sky, c2 = gold light, c3 = rose horizon glow (palette slots 1-3).
void main() {
    vec2 uv = isf_FragNormCoord;
    float ph = TIME;
    // The bow: the horizon pivots about the centre by `bow` (-1..1), which a
    // value curve drives to alternate extremes on each bow change -- a
    // POSITION, so it cannot drift the way an integrated clock does.
    float tilt = sway * bow;
    float swell = 0.025 * sin(uv.x * 9.4 + ph * 0.6);
    float h = height + tilt * (uv.x - 0.5) * 2.0 + swell;
    float d = uv.y - h;                                   // > 0 is sky
    vec3 col = mix(c2.rgb, c1.rgb, smoothstep(-soft, soft, d));
    // the rose band sits on the horizon line and moves with it
    col = mix(col, c3.rgb, 0.85 * exp(-pow(d / (soft * 0.7), 2.0)));
    // faint gold rays fanning up from below the horizon
    float ang = atan(uv.x - 0.5, d + 0.35);
    float r = 0.5 + 0.5 * sin(ang * 16.0 + ph * 0.35);
    col += c2.rgb * rays * r * smoothstep(0.02, 0.25, d) * (1.0 - smoothstep(0.25, 0.9, d));
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
