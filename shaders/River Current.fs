/*{
  "CREDIT": "written for the Auders 2026 show (Jackson Lights 2026 rev3, Gregory Porter - Liquid Spirit, Claptone remix)",
  "CATEGORIES": ["2d", "generator", "organic"],
  "DESCRIPTION": "'Un-dam the rivers': broad meandering channels of water flowing sideways across the canvas, the deep colour between them, with bright current streaks running along each channel. Field-filling and even -- every part of the canvas looks like every other part, so ten matrices running it read as one river. `rate` is signed: the sway reverses the current on the beat.",
  "INPUTS": [
    { "NAME": "rate",    "TYPE": "float", "DEFAULT": 0.5,  "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "bands",   "TYPE": "float", "DEFAULT": 3.0,  "MIN": 1.0,  "MAX": 8.0 },
    { "NAME": "meander", "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 1.5 },
    { "NAME": "streaks", "TYPE": "float", "DEFAULT": 0.5,  "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "c1", "TYPE": "color", "DEFAULT": [0.0, 0.55, 0.9, 1.0] },
    { "NAME": "c2", "TYPE": "color", "DEFAULT": [1.0, 0.8, 0.35, 1.0] },
    { "NAME": "c3", "TYPE": "color", "DEFAULT": [0.05, 0.1, 0.4, 1.0] }
  ]
}*/
// c1 = the water, c2 = the current streaks, c3 = the deep between channels.
float hr(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float vnr(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hr(i), hr(i + vec2(1, 0)), f.x), mix(hr(i + vec2(0, 1)), hr(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * rate;
    // channels meander: the vertical position bends with slow noise along x
    float bend = meander * (vnr(vec2(uv.x * 2.0 - t * 0.15, 3.1)) - 0.5)
               + 0.25 * meander * sin(uv.x * 5.0 + t * 0.4);
    float y = (uv.y + bend * 0.35) * bands;
    float c = fract(y);
    // a channel is a soft band in the middle of each period
    float water = smoothstep(0.08, 0.3, c) * smoothstep(0.92, 0.7, c);
    // current streaks: thin bright dashes flowing along x inside the channel
    float lane = floor(y);
    float flow = uv.x * 6.0 - t * (1.2 + 0.6 * hr(vec2(lane, 1.7)));
    float dash = smoothstep(0.55, 0.95, vnr(vec2(flow * 1.5, lane * 7.3 + c * 3.0)));
    float centre = 1.0 - abs(c - 0.5) * 2.0;
    float streak = dash * smoothstep(0.3, 0.9, centre) * streaks;
    // depth shading across the channel so it reads as rounded water
    vec3 col = mix(c3.rgb, c1.rgb * (0.6 + 0.4 * centre), water);
    col = mix(col, c2.rgb, clamp(streak, 0.0, 1.0));
    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
