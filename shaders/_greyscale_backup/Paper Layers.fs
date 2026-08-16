/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "Cut-paper diorama: soft-edged hills stacked in depth, each layer sliding at its own parallax speed with a shadow cast on the one behind. Reads as physical depth on a flat wall.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "layers",  "TYPE": "float", "DEFAULT": 5.0, "MIN": 2.0,  "MAX": 9.0 },
    { "NAME": "relief",  "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "rolling", "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.1,  "MAX": 1.5 }
  ]
}*/
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
    gl_FragColor = vec4(vec3(clamp(v,0.0,1.0)), clamp(v,0.0,1.0));
}
