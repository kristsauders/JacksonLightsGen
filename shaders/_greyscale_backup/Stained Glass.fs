/*{
  "CREDIT": "written for the Auders 2026 show",
  "CATEGORIES": ["2d", "generator", "geometric"],
  "DESCRIPTION": "Leaded glass: irregular panes each holding a steady tone, separated by dark came, with light moving slowly behind so panes brighten in turn. Architectural rather than digital.",
  "INPUTS": [
    { "NAME": "rate",  "TYPE": "float", "DEFAULT": 0.22, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "panes", "TYPE": "float", "DEFAULT": 5.0,  "MIN": 2.0,  "MAX": 14.0 },
    { "NAME": "came",  "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.1,  "MAX": 1.0 },
    { "NAME": "behind","TYPE": "float", "DEFAULT": 0.7,  "MIN": 0.0,  "MAX": 1.0 }
  ]
}*/
vec2 hs2(vec2 p){
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    vec2 p = uv * panes;
    vec2 ip = floor(p), fp = fract(p);
    float d1 = 8.0, d2 = 8.0; vec2 id = vec2(0.0);
    for (int j = -1; j <= 1; j++) {
      for (int i = -1; i <= 1; i++) {
        vec2 g = vec2(float(i), float(j));
        vec2 h = hs2(ip + g);
        vec2 o = g + h;                      // static seeds: panes don't move
        float d = length(o - fp);
        if (d < d1) { d2 = d1; d1 = d; id = ip + g; }
        else if (d < d2) { d2 = d; }
      }
    }
    // Each pane holds its own steady tone.
    float tone = 0.35 + 0.65 * hs2(id).x;
    // Light passing behind the window lifts panes in turn.
    float sweep = 0.5 + 0.5 * sin(dot(id, vec2(0.7, 0.4)) - t * 1.6);
    tone *= mix(1.0, 0.45 + 0.75 * sweep, behind);
    // Dark came between panes.
    float lead = smoothstep(0.0, came * 0.14, d2 - d1);
    float v = tone * lead;
    gl_FragColor = vec4(vec3(clamp(v,0.0,1.0)), clamp(v,0.0,1.0));
}
