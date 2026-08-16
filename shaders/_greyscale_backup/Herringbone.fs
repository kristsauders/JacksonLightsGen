/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "texture"],
  "DESCRIPTION": "Woven herringbone twill. A textile texture with no focal point, so ten props can run it at once and it reads as one fabric rather than ten copies of a picture.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.3, "MIN": -1.2, "MAX": 1.2 },
    { "NAME": "scale", "TYPE": "float", "DEFAULT": 7.0, "MIN": 2.0,  "MAX": 20.0 },
    { "NAME": "slant", "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.3,  "MAX": 2.5 },
    { "NAME": "weft",  "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    vec2 p = uv * scale;
    float band = floor(p.y);
    float dir = mod(band, 2.0) < 1.0 ? 1.0 : -1.0;      // alternate the twill
    float d = p.x * slant * dir + p.y + t;
    float w = fract(d);
    float v = smoothstep(0.0, 0.42, w) * smoothstep(1.0, 0.58, w);
    // Weft threads crossing the warp give it cloth grain.
    float cross_ = 0.5 + 0.5 * sin(p.y * 6.28318);
    v = mix(v, v * (0.55 + 0.45 * cross_), weft);
    gl_FragColor = vec4(vec3(clamp(v, 0.0, 1.0)), clamp(v, 0.0, 1.0));
}
