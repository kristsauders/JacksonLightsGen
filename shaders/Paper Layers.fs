/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "Cut-paper diorama: soft-edged hills stacked in depth, each layer sliding at its own parallax speed with a shadow cast on the one behind. Reads as physical depth on a flat wall.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "layers",  "TYPE": "float", "DEFAULT": 5.0, "MIN": 2.0,  "MAX": 9.0 },
    { "NAME": "relief",  "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "rolling", "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.1,  "MAX": 1.5 }
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
// Where in the palette this pixel sits: a slow diagonal drift, nudged by the
// shader's own value so bright detail still leans toward the palette's top
// end. Weighted toward the drift because value alone is invisible -- alpha
// hides exactly the part of the ramp that value would have shown.
float palpos(vec2 uv, float v) {
    float drift = 0.5 + 0.5 * sin((uv.x * 1.6 + uv.y * 1.1) * 2.1 - TIME * 0.11);
    return clamp(mix(drift, v, 0.35), 0.0, 1.0);
}

void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    float v = 0.0;
    for (int i = 0; i < 9; i++) {
        if (float(i) >= layers) break;
        float fi = float(i);
        float depth = fi / max(layers - 1.0, 1.0);      // 0 = far, 1 = near
        // Nearer layers slide faster: parallax is the whole illusion.
        float px = uv.x + t * (0.05 + 0.22 * depth) + fi * 0.37;
        // A silhouette built from a few sines -- deliberately smooth, like a
        // knife cut, not noise.
        float ridge = 0.10 + 0.42 * depth
                    + rolling * 0.11 * sin(px * 4.1 + fi)
                    + rolling * 0.06 * sin(px * 9.3 - fi * 2.0);
        float inside = 1.0 - smoothstep(ridge, ridge + 0.006, uv.y);
        // Near layers sit brighter (closer to the light), and each casts a
        // soft shadow just above its own edge onto whatever is behind.
        float face = mix(0.30, 1.0, depth);
        float shadow = 1.0 - relief * 0.55 * exp(-max(uv.y - ridge, 0.0) / 0.055);
        v = mix(v * shadow, face, inside);
    }
    float _pv = clamp(v,0.0,1.0);
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
