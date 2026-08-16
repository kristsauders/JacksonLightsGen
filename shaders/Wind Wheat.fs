/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "A field of stalks bending under gusts that travel across the canvas. The gust is a visible wave passing through, not every stalk swaying in unison -- that is the whole point.",
  "INPUTS": [
    { "NAME": "rate",   "TYPE": "float", "DEFAULT": 0.4, "MIN": -1.5, "MAX": 1.5 },
    { "NAME": "stalks", "TYPE": "float", "DEFAULT": 40.0,"MIN": 8.0,  "MAX": 90.0 },
    { "NAME": "gust",   "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "height", "TYPE": "float", "DEFAULT": 0.8, "MIN": 0.3,  "MAX": 1.0 }
  ,
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [1.0, 1.0, 1.0, 1.0] }
  ]
}*/
float hw(float n){ return fract(sin(n * 91.17) * 43758.5453); }
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
    float _pv = clamp(v,0.0,1.0);
    gl_FragColor = vec4(palcol(palpos(isf_FragNormCoord, _pv)), _pv);
}
