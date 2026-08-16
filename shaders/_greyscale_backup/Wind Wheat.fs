/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "A field of stalks bending under gusts that travel across the canvas. The gust is a visible wave passing through, not every stalk swaying in unison -- that is the whole point.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.4, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "stalks", "TYPE": "float", "DEFAULT": 40.0,"MIN": 8.0,  "MAX": 90.0 },
    { "NAME": "gust",   "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "height", "TYPE": "float", "DEFAULT": 0.8, "MIN": 0.3,  "MAX": 1.0 }
  ]
}*/
float hw(float n){ return fract(sin(n * 91.17) * 43758.5453); }
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    // A gust is a travelling envelope: strong in a band, calm either side.
    float gpos = fract(t * 0.25);
    float band = exp(-pow((uv.x - gpos) * 2.6, 2.0)) + exp(-pow((uv.x - gpos + 1.0) * 2.6, 2.0));
    float strength = gust * (0.35 + 0.9 * band);
    float col = uv.x * stalks;
    float v = 0.0;
    // Sample the two nearest stalk columns so bending stays smooth.
    for (int k = 0; k <= 1; k++) {
        float ci = floor(col) + float(k);
        float seed = hw(ci);
        float h = height * (0.6 + 0.4 * seed);
        // Bend grows with height above the root -- stalks pivot at the base.
        float up = clamp(uv.y / max(h, 0.01), 0.0, 1.0);
        float sway = sin(t * (1.6 + seed) + ci * 0.7) * strength * 0.09 * up * up;
        float x = (ci + 0.5) / stalks + sway;
        float d = abs(uv.x - x) * stalks;
        float body = (1.0 - smoothstep(0.12, 0.5, d)) * step(uv.y, h);
        // Heads catch more light than stems.
        float head = smoothstep(h * 0.72, h, uv.y);
        v = max(v, body * (0.45 + 0.55 * head));
    }
    gl_FragColor = vec4(vec3(clamp(v,0.0,1.0)), clamp(v,0.0,1.0));
}
