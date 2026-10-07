"""Jackson Lights 2026 -- the launch-party medley. 28:49, fifteen songs.

The show's design, song by song, in MIX time (what you read off the track in
xLights). medley_prep.py converts it into each song's medley.json.

How the songs were identified: Shazam fingerprints every 10 s (most songs),
Whisper lyrics (to place vocals), and Krists for the three Shazam could not
name -- the Nils Frahm opener, Spectacle's "Every Night", and the J3 /
eXecutive Decision / Kevin Aleksander "Deck the Halls" remix.

How the boundaries were found: a whole-mix madmom beat+downbeat grid (which
follows the DJ's tempo ramps), the tempo curve read off it, and Shazam's
song-change points; each cut sits inside the crossfade on a real downbeat.
Sections inside each song are hand-read from section_plan.py's four-bar
phrase map -- the structure detector is not trusted (see runbook 3b).

The brief, in Krists's words:
  * song 1 "very low illumination, like 10-20%... the warmup telling people
    the show is starting"
  * song 2 "starts with an organ sound... still pretty low illumination like
    maybe 30-50% until the first big drop"
  * "6 rapid hits spaced evenly, then a brief break, then a drop... 100%
    illuminated but with color contrast... the big reveal of the whole display"
  * "give each song its own unique look... don't re-use the same effects and
    shaders too much across songs, make it one big masterpiece"
  * "the ending is something beautiful and flowing because it's a string
    orchestra outro... movement but not flashing and then a slow fadeout"

SHADERS: the 50 registered shaders are split across the 15 songs with NO
shader used by two songs. Each pool is chosen for its song's mood.

WHICH SHADER GOES WHERE is decided by MEASURED brightness, not by name. A
shader's own content sets how bright a section reads far more than the
brightness curve does: on the first full render nine songs had a chorus or
drop dimmer than the verse or break around it (greedy's choruses 0.27 against
a 0.47 intro; Glad I Found You's choruses ~0.23 against builds at 0.40; the
reveal chorus 0.37 on Black Cherry Cosmos). Within each song the brightest
measured shader now carries the drops and choruses.
"""

DESIGN: dict[str, dict] = {

    # =====================================================================
    # 01  Nils Frahm -- Ode            0:00 - 1:35.9          ~62 BPM, no beat
    # "A beautifully evocative, calming piece that feels like sinking into a
    # warm dream on a hazy summer evening." So the warm-up is warm amber haze,
    # not cold blue, held to 10-20% the whole way and sequenced as SHAPE
    # (ambient), not beat. Phrase map: soft until 0:18, swells at 0:33 (bass
    # in), fullest at 1:04, thinning from 1:19.
    # =====================================================================
    "01 Ode Our Own Roof": {
        "palettes": {"hearth": ["#FF9A3C", "#FF6F61", "#FFE3B3"],
                     "haze":   ["#FFB07C", "#C97B63", "#FFF1D6"]},
        "palette_family": ["hearth", "haze"],
        "shader_pool": ["slow_dawn", "bokeh_drift", "smoke_drift"],
        "sections": [(0.0, "intro"), (33.12, "verse"), (62.92, "chorus"), (79.12, "outro")],
        "section_look": {
            0.0:   {"palette": "haze",   "shader": "shader_bokeh_drift"},
            33.12: {"palette": "hearth", "shader": "shader_smoke_drift"},
            62.92: {"palette": "hearth", "shader": "shader_slow_dawn"},
            79.12: {"palette": "haze",   "shader": "shader_bokeh_drift"},
        },
        "ambient": [(0.0, 95.89)],
        "caps": [(0.0, 105.89, 0.10, 0.20)],
    },

    # =====================================================================
    # 02  Muse -- Be With You          1:35.9 - 4:10.7        organ, then 130
    # Church pipe organ (recorded in a Los Angeles cathedral, "reminiscent of
    # Interstellar"), then electronic, then stadium space rock -- from an album
    # called The Wow! Signal. So the arc is CATHEDRAL -> SPACE: stained glass
    # in violet/gold/crimson for the organ, a cosmic cyan/magenta after the
    # drop. Held at 30-50% until the reveal.
    #   1:35.9 organ in   1:43.5 light percussion   2:43.6 guitar build
    #   2:51.2 bass under the build   2:58.16-2:59.29 THE SIX HITS (drum stem)
    #   2:59.73 THE DROP   3:13.7 lift   3:58.8 breakdown   4:06.4 rise
    # =====================================================================
    "02 Be With You": {
        "palettes": {"cathedral": ["#6A1FB0", "#E8B54A", "#9E1B32"],
                     "wowsignal": ["#00E5FF", "#FF2BD6", "#FFFFFF"],
                     "nebula":    ["#3A0CA3", "#00E5FF", "#FFD700"]},
        "palette_family": ["cathedral", "wowsignal", "nebula"],
        "shader_pool": ["stained_glass", "black_cherry_cosmos", "space_spinning", "hex_3d_spiral"],
        "sections": [(95.89, "intro"), (103.20, "verse"), (163.47, "build"),
                     (171.26, "build"), (179.73, "chorus"), (193.71, "chorus"),
                     (239.04, "break"), (245.49, "build")],
        "section_look": {
            95.89:  {"palette": "cathedral", "shader": "shader_stained_glass"},
            103.20: {"palette": "cathedral", "shader": "shader_stained_glass"},
            163.47: {"palette": "nebula",    "shader": "shader_hex_3d_spiral"},
            171.26: {"palette": "nebula",    "shader": "shader_space_spinning"},
            179.73: {"palette": "wowsignal", "shader": "shader_hex_3d_spiral"},
            193.71: {"palette": "wowsignal", "shader": "shader_stained_glass"},
            239.04: {"palette": "cathedral", "shader": "shader_stained_glass"},
            245.49: {"palette": "nebula",    "shader": "shader_black_cherry_cosmos"},
        },
        "ambient": [(95.89, 163.47)],
        "caps": [(85.89, 178.13, 0.30, 0.50)],
        "reveal": {"hits": [178.160, 178.375, 178.594, 178.829, 179.053, 179.288],
                   "drop": 179.732, "colours": ["#FFD700", "#0047FF"]},
        "drops": [179.732],
        # The reveal chorus wears the BRIGHTEST shader of the song. Black Cherry
        # Cosmos is mostly dark sky: measured on the first build, the chorus
        # straight after the reveal sat at display mean 0.37 while the second
        # chorus, on Hex 3D Spiral, reached 0.54 -- the launch moment dimmer
        # than an ordinary chorus.
        # A second INTO the chorus, not on the drop: the generator snaps the
        # chorus boundary onto a beat (it landed 0.04s after the drop), and a
        # climax marker exactly on the drop then fell into the BUILD -- which
        # took 100% while the reveal chorus got 85%.
        "climax": [180.75],
    },

    # =====================================================================
    # 03  Bluey Dance Mode (Zeeslow remix)   4:10.7 - 4:50.4   127.7 BPM
    # A 40-second edit: two short drop cycles. Bluey's blue and Bingo's orange,
    # cartoon-flat shaders. Groove mode, as decided for the full-length version.
    # =====================================================================
    "03 Bluey Dance Mode": {
        "palettes": {"bluey": ["#2E8BFF", "#FF8C1A", "#FFFFFF"],
                     "bingo": ["#FF8C1A", "#FFD27A", "#2E8BFF"]},
        "palette_family": ["bluey", "bingo"],
        "shader_pool": ["halftone_pulse", "sequin_flip", "zoetrope_slats"],
        "sections": [(250.69, "intro"), (256.36, "break"), (263.93, "chorus"),
                     (271.01, "break"), (279.04, "chorus")],
        "section_look": {
            250.69: {"palette": "bluey", "shader": "shader_sequin_flip"},
            256.36: {"palette": "bingo", "shader": "shader_zoetrope_slats"},
            263.93: {"palette": "bluey", "shader": "shader_halftone_pulse"},
            271.01: {"palette": "bingo", "shader": "shader_zoetrope_slats"},
            279.04: {"palette": "bluey", "shader": "shader_halftone_pulse"},
        },
        "drops": [263.93, 279.04],
        "groove": True,
    },
    # =====================================================================
    # 04  sombr -- 12 to 12            4:50.4 - 7:10.0       125 BPM
    # Disco-pop with a lovesick croon: MIDNIGHT DISCO -- magenta, teal and
    # mirror-ball silver, on op-art shaders (moire, corduroy ridges, the
    # metablob starfield). The DJ dips the bass out at 4:52-4:58 and the
    # song lands at 5:00.1 (+0.42, the biggest step in the segment).
    #   5:00.1 "I don't want anyone else"   5:30.9 second verse half
    #   5:46.2 pre-chorus (hats climbing)   6:01.6 "In a room full of people"
    #   6:32.3 verse 2   6:47.7 "Paris cafe" lift   7:03.0 chorus tail
    # =====================================================================
    "04 12 to 12": {
        "palettes": {"disco":      ["#FF2E88", "#00C2B8", "#E6E6F0"],
                     "mirrorball": ["#B388FF", "#FF2E88", "#FFFFFF"],
                     "afterhours": ["#00C2B8", "#3D1A78", "#FF2E88"]},
        "palette_family": ["disco", "mirrorball", "afterhours"],
        "shader_pool": ["moire_rings", "metablob_starz", "corduroy"],
        "sections": [(290.38, "break"), (300.29, "verse"), (330.78, "verse"),
                     (346.21, "build"), (361.75, "chorus"), (392.23, "verse"),
                     (407.72, "build"), (423.20, "chorus")],
        "section_look": {
            290.38: {"palette": "afterhours", "shader": "shader_metablob_starz"},
            300.29: {"palette": "disco",      "shader": "shader_corduroy"},
            330.78: {"palette": "afterhours", "shader": "shader_corduroy"},
            346.21: {"palette": "mirrorball", "shader": "shader_metablob_starz"},
            361.75: {"palette": "disco",      "shader": "shader_moire_rings"},
            392.23: {"palette": "afterhours", "shader": "shader_corduroy"},
            407.72: {"palette": "mirrorball", "shader": "shader_metablob_starz"},
            423.20: {"palette": "disco",      "shader": "shader_moire_rings"},
        },
        "drops": [300.29, 361.75],
        # A climax is named by its SECTION START; medley_prep puts the marker
        # a second inside (a marker exactly on a boundary can land in the
        # section before -- see 02).
        "climax": [361.75],
    },

    # =====================================================================
    # 05  Tate McRae -- greedy         7:10.0 - 8:50.3       111 BPM
    # Brash, glossy, dance-pop: HOT PINK and lipstick red with white, on
    # slick shaders (liquid chrome, chromatophore skin, iso blocks).
    #   7:17.6 verse "I see you"   7:34.9 chorus "Baby please believe me"
    #   7:52.1 verse 2 "You said I'm just curious"   8:09.4 chorus
    #   8:26.7 post-chorus vamp   8:44.0 thinning into the Santa trap
    # =====================================================================
    "05 greedy": {
        "palettes": {"greedy":   ["#FF1493", "#E10600", "#FFFFFF"],
                     "lipstick": ["#C2185B", "#FF5C8A", "#FFD1DC"]},
        "palette_family": ["greedy", "lipstick"],
        "shader_pool": ["liquid_chrome", "chromatophore", "iso_blocks"],
        "sections": [(429.99, "intro"), (437.39, "verse"), (455.13, "chorus"),
                     (472.30, "verse"), (490.30, "chorus"), (506.70, "break"),
                     (525.21, "outro")],
        "section_look": {
            429.99: {"palette": "greedy",   "shader": "shader_liquid_chrome"},
            437.39: {"palette": "lipstick", "shader": "shader_chromatophore"},
            455.13: {"palette": "greedy",   "shader": "shader_iso_blocks"},
            472.30: {"palette": "lipstick", "shader": "shader_chromatophore"},
            490.30: {"palette": "greedy",   "shader": "shader_iso_blocks"},
            506.70: {"palette": "lipstick", "shader": "shader_liquid_chrome"},
            525.21: {"palette": "lipstick", "shader": "shader_chromatophore"},
        },
        "drops": [455.13, 490.30],
        "climax": [490.30],
    },

    # =====================================================================
    # 06  StarBoy -- Here Comes Santa Claus (trap remix)   8:50.3 - 9:40.5
    # 105 BPM half-time trap, flat energy throughout (no phrase change point
    # reaches the threshold), Gene Autry's vocal on top. The first proper
    # CHRISTMAS colours of the night -- red, green, white -- on knitwear
    # shaders (argyle, tartan, cable knit): the ugly-sweater song. Groove
    # pulse: kick + snare, which is how trap is felt.
    #   8:55.1 "Here comes Santa Claus"   9:13.3 second verse
    #   9:31.6 Gene Autry original under the handover to Elderbrook
    # =====================================================================
    "06 Here Comes Santa Claus": {
        "palettes": {"sweater": ["#D10000", "#00A651", "#FFFFFF"],
                     "sleigh":  ["#00A651", "#FFFFFF", "#D10000"]},
        "palette_family": ["sweater", "sleigh"],
        "shader_pool": ["argyle_march", "tartan_shift", "cable_knit"],
        "sections": [(530.28, "intro"), (534.82, "chorus"), (553.11, "chorus"),
                     (571.40, "break")],
        "section_look": {
            530.28: {"palette": "sleigh",  "shader": "shader_cable_knit"},
            534.82: {"palette": "sweater", "shader": "shader_tartan_shift"},
            553.11: {"palette": "sleigh",  "shader": "shader_argyle_march"},
            571.40: {"palette": "sweater", "shader": "shader_cable_knit"},
        },
        "drops": [534.82],
        "pulse_mode": "groove",
    },
    # =====================================================================
    # 07  Elderbrook & George FitzGerald -- Glad I Found You   9:40.5 - 11:41.7
    # Warm, euphoric deep house at 120: CORAL and peach against teal -- a
    # sunset over water -- on soft, organic movers (wind through wheat,
    # caustics on a pool floor, braided ribbons). Four drops, each after a
    # one- or two-bar dip; the biggest (+0.41) follows the near-silent
    # breakdown at 10:53, so that one is the song's climax.
    #   9:49.4 drop   10:03.4 "Come on, say"   10:17.4 dip
    #   10:21.4 "Fly, fly, fly to you"   10:51.4 breakdown (silence 10:53)
    #   10:57.4 drop   11:25.4 dip   11:29.4 "Fly like a new"
    # =====================================================================
    "07 Glad I Found You": {
        "palettes": {"sunset": ["#FF6F59", "#FFB38A", "#13A89E"],
                     "lagoon": ["#13A89E", "#7FDBDA", "#FF8C69"]},
        "palette_family": ["sunset", "lagoon"],
        "shader_pool": ["wind_wheat", "caustic_pool", "ribbon_braid"],
        "sections": [(580.54, "intro"), (589.65, "chorus"), (603.65, "verse"),
                     (617.64, "build"), (621.65, "chorus"), (653.64, "break"),
                     (657.65, "chorus"), (685.64, "build"), (689.65, "chorus")],
        "section_look": {
            580.54: {"palette": "lagoon", "shader": "shader_ribbon_braid"},
            589.65: {"palette": "sunset", "shader": "shader_caustic_pool"},
            603.65: {"palette": "lagoon", "shader": "shader_wind_wheat"},
            617.64: {"palette": "lagoon", "shader": "shader_ribbon_braid"},
            621.65: {"palette": "sunset", "shader": "shader_caustic_pool"},
            653.64: {"palette": "lagoon", "shader": "shader_wind_wheat"},
            657.65: {"palette": "sunset", "shader": "shader_caustic_pool"},
            685.64: {"palette": "lagoon", "shader": "shader_ribbon_braid"},
            689.65: {"palette": "sunset", "shader": "shader_caustic_pool"},
        },
        "drops": [589.65, 621.65, 657.65, 689.65],
        "climax": [657.65],
    },
    # =====================================================================
    # 08  Bad Bunny -- NUEVAYoL         11:41.7 - 13:00.2     117.7 BPM
    # A love letter to New York's Puerto Ricans, opening on a salsa sample:
    # the flag's RED, BLUE and white, with taxi-cab yellow for the city, on
    # urban/structural shaders (circuit trace, herringbone brick, fracture).
    # Groove pulse -- it is dembow, felt on kick and snare, not a metronome.
    #   11:45.9 the room goes quiet   11:50.0 salsa sample intro
    #   12:04.3 the beat lands (bass 0.82 -> 1.36)   12:20.6 rap verse
    #   12:36.9 second verse   12:55.3 bass out -- silence into ENIGMA
    # =====================================================================
    "08 NUEVAYoL": {
        "palettes": {"boricua":  ["#E4002B", "#0050F0", "#FFFFFF"],
                     "nuevayol": ["#FFC72C", "#E4002B", "#0050F0"]},
        "palette_family": ["boricua", "nuevayol"],
        "shader_pool": ["circuit_trace", "herringbone", "fracture"],
        "sections": [(701.65, "outro"), (706.14, "break"), (709.85, "intro"),
                     (724.09, "chorus"), (741.89, "verse"), (757.14, "verse"),
                     (775.04, "break")],
        "section_look": {
            701.65: {"palette": "boricua",  "shader": "shader_herringbone"},
            706.14: {"palette": "boricua",  "shader": "shader_herringbone"},
            709.85: {"palette": "nuevayol", "shader": "shader_herringbone"},
            724.09: {"palette": "boricua",  "shader": "shader_fracture"},
            741.89: {"palette": "nuevayol", "shader": "shader_circuit_trace"},
            757.14: {"palette": "boricua",  "shader": "shader_fracture"},
            775.04: {"palette": "nuevayol", "shader": "shader_circuit_trace"},
        },
        "drops": [724.09],
        "climax": [724.09],
        "pulse_mode": "groove",
    },
    # =====================================================================
    # 09  CloZee & LSDREAM -- ENIGMA    13:00.2 - 15:08.9     96.8 BPM
    # Tribal world-bass: a chant over near-silence, then heavy half-time
    # bass. The mix's MID-SHOW BREATH -- the opening 25 s is ambient -- then
    # the darkest, strangest look of the night: ACID GREEN on deep violet,
    # with ritual-fire orange, on organic/uncanny shaders (murmuration,
    # ink bloom, interference, hex vortex).
    #   13:00.2 chant over silence   13:25.6 bass creeps in
    #   13:28.1 drop   13:42.9 two-bar hat break   13:47.9 drop
    #   14:07.8 breakdown (chant)   14:22.6 build   14:27.6 BIGGEST drop
    #   14:42.5 second half   15:04.8 hats-only tail into Every Night
    # =====================================================================
    "09 ENIGMA": {
        "palettes": {"enigma": ["#7CFF00", "#5B0E91", "#00FFC6"],
                     "ritual": ["#5B0E91", "#B6FF3B", "#FF7A00"]},
        "palette_family": ["enigma", "ritual"],
        "shader_pool": ["murmuration", "ink_bloom", "interference", "hex_vortex"],
        "sections": [(780.23, "intro"), (807.40, "chorus"),
                     (824.11, "break"), (827.20, "chorus"), (847.66, "break"),
                     (862.54, "build"), (866.80, "chorus"), (882.26, "verse"),
                     (904.54, "outro")],
        "section_look": {
            780.23: {"palette": "ritual", "shader": "shader_murmuration"},
            807.40: {"palette": "enigma", "shader": "shader_hex_vortex"},
            824.11: {"palette": "ritual", "shader": "shader_interference"},
            827.20: {"palette": "enigma", "shader": "shader_ink_bloom"},
            847.66: {"palette": "ritual", "shader": "shader_murmuration"},
            862.54: {"palette": "ritual", "shader": "shader_interference"},
            866.80: {"palette": "enigma", "shader": "shader_hex_vortex"},
            882.26: {"palette": "ritual", "shader": "shader_ink_bloom"},
            904.54: {"palette": "enigma", "shader": "shader_interference"},
        },
        "ambient": [(780.23, 805.59)],
        "drops": [807.40, 827.20, 866.80],
        "climax": [866.80],
    },
    # =====================================================================
    # 10  Spectacle -- Every Night      15:08.9 - 18:31.3     100 BPM
    # Shazam could not name it (Krists did). Night-drive electronic: the
    # longest song in the mix, so the look must evolve without leaving its
    # identity -- INDIGO and ultraviolet with electric cyan, on slow
    # contour/tidal shaders that suit a night scene (drift cells, contour
    # drift, tide lines, pulse rings). Its 15 s breakdown is the mix's
    # second breath.
    #   15:08.9 busy hats-heavy opening   15:39.8 "Every night"
    #   16:03.8 thins   16:13.4 breakdown (ambient)   16:27.8 bass returns
    #   16:39.8 verse   17:01.4 dip   17:11.0 "I am the day" (step up)
    #   17:30.2 "J-I-N-G-L-E"   17:49.4 lift to the peak   18:25.4 outro
    # =====================================================================
    "10 Every Night": {
        "palettes": {"nocturne":  ["#2B1B8F", "#8A2BE2", "#00D4FF"],
                     "afterglow": ["#00D4FF", "#6A00FF", "#FF5EDB"]},
        "palette_family": ["nocturne", "afterglow"],
        "shader_pool": ["drift_cells", "contour_drift", "tide_lines", "pulse_rings"],
        "sections": [(908.86, "verse"), (939.54, "chorus"), (963.65, "break"),
                     (973.69, "break"), (988.04, "build"), (999.48, "verse"),
                     (1020.69, "break"), (1030.39, "chorus"), (1050.39, "verse"),
                     (1069.18, "chorus"), (1106.12, "outro")],
        "section_look": {
            908.86:  {"palette": "nocturne",  "shader": "shader_pulse_rings"},
            939.54:  {"palette": "afterglow", "shader": "shader_drift_cells"},
            963.65:  {"palette": "nocturne",  "shader": "shader_tide_lines"},
            973.69:  {"palette": "nocturne",  "shader": "shader_contour_drift"},
            988.04:  {"palette": "afterglow", "shader": "shader_tide_lines"},
            999.48:  {"palette": "nocturne",  "shader": "shader_pulse_rings"},
            1020.69: {"palette": "nocturne",  "shader": "shader_contour_drift"},
            1030.39: {"palette": "afterglow", "shader": "shader_drift_cells"},
            1050.39: {"palette": "nocturne",  "shader": "shader_tide_lines"},
            1069.18: {"palette": "afterglow", "shader": "shader_drift_cells"},
            1106.12: {"palette": "nocturne",  "shader": "shader_contour_drift"},
        },
        "ambient": [(973.69, 988.04)],
        "drops": [1030.39],
        "climax": [1069.18],
    },
    # =====================================================================
    # 11  Bing Crosby -- Happy Holidays (Beef Wellington remix)  18:31.3 - 20:10.3
    # A 1940s crooner over a modern house beat at 92: the show's VINTAGE
    # moment -- antique gold, cream and cranberry with a little pine, on
    # luxurious material shaders (velvet nap, oil film, marble veining).
    # Opens soft (ambient) on the crooner, the beat drops at 18:49.4, then
    # eight-bar phrases.
    #   18:36.4 near silence   18:49.4 drop "Happy holiday"
    #   19:10.3 verse   19:31.2 chorus   19:49.5 one-bar dip   19:52.1 last chorus
    #   20:07.7 handover to Santa Claus Is Coming to Town
    # =====================================================================
    "11 Happy Holidays": {
        "palettes": {"fireside": ["#A4123F", "#D4A017", "#FFF3D6"],
                     "tinsel":   ["#D4A017", "#FFF3D6", "#2E6B3F"]},
        "palette_family": ["fireside", "tinsel"],
        "shader_pool": ["velvet_nap", "oil_film", "marble_vein"],
        "sections": [(1111.28, "intro"), (1127.47, "build"), (1129.44, "chorus"),
                     (1149.67, "verse"), (1171.19, "chorus"), (1189.45, "break"),
                     (1192.06, "chorus"), (1207.70, "outro")],
        "section_look": {
            1111.28: {"palette": "tinsel",   "shader": "shader_velvet_nap"},
            1127.47: {"palette": "tinsel",   "shader": "shader_oil_film"},
            1129.44: {"palette": "fireside", "shader": "shader_oil_film"},
            1149.67: {"palette": "tinsel",   "shader": "shader_marble_vein"},
            1171.19: {"palette": "fireside", "shader": "shader_oil_film"},
            1189.45: {"palette": "tinsel",   "shader": "shader_velvet_nap"},
            1192.06: {"palette": "fireside", "shader": "shader_oil_film"},
            1207.70: {"palette": "tinsel",   "shader": "shader_velvet_nap"},
        },
        "ambient": [(1111.28, 1127.47)],
        "drops": [1129.44],
        "climax": [1192.06],
    },
    # =====================================================================
    # 12  Dimitri Vegas & Like Mike, R3HAB -- Santa Claus Is Coming to Town
    #     20:10.3 - 21:20.0     125.5 BPM
    # Festival big-room Christmas: CANDY CANE red and white with peppermint,
    # on bold symmetric shaders (kaleido bloom, target circles, the
    # inner-dimensional matrix). Two drops, the second after a one-bar
    # silence with the heaviest bass in the mix (2.10).
    #   20:10.3 "You better watch out"   20:28.3 bass in   20:39.8 riser
    #   20:43.6 DROP   20:59.0 second half   21:12.4 silence   21:14.3 DROP 2
    # =====================================================================
    "12 Santa Claus Is Coming to Town": {
        "palettes": {"candycane":  ["#FF0A3C", "#FFFFFF", "#FF7A9A"],
                     "peppermint": ["#FFFFFF", "#FF0A3C", "#00B37A"]},
        "palette_family": ["candycane", "peppermint"],
        "shader_pool": ["kaleido_bloom", "target_circle", "innerdimensionalmatrix"],
        "sections": [(1210.32, "verse"), (1228.27, "build"), (1239.78, "break"),
                     (1243.61, "chorus"), (1258.98, "chorus"), (1272.40, "break"),
                     (1274.34, "chorus")],
        "section_look": {
            1210.32: {"palette": "peppermint", "shader": "shader_target_circle"},
            1228.27: {"palette": "peppermint", "shader": "shader_kaleido_bloom"},
            1239.78: {"palette": "candycane",  "shader": "shader_target_circle"},
            1243.61: {"palette": "candycane",  "shader": "shader_innerdimensionalmatrix"},
            1258.98: {"palette": "peppermint", "shader": "shader_kaleido_bloom"},
            1272.40: {"palette": "candycane",  "shader": "shader_target_circle"},
            1274.34: {"palette": "candycane",  "shader": "shader_innerdimensionalmatrix"},
        },
        "drops": [1243.61, 1274.34],
        "climax": [1243.61],
    },
    # =====================================================================
    # 13  J3, eXecutive Decision, Kevin Aleksander -- Deck the Halls
    #     21:20.0 - 24:15.5     128.5 BPM
    # Heavy dance Deck the Halls (Shazam could not name it; Krists did), and
    # from 23:03 a "Last Christmas" vocal over the same groove (Whisper heard
    # the lyrics; Shazam misread it as sleep music) -- the DJ's blend. Two
    # looks inside one song: a roaring HOLLY-and-gold hearth for the drops
    # (ember swarm), frost already creeping over the second half of the long
    # drop (22:36.9), then FROST for Last Christmas -- warm to cold, the heart
    # given away. (Frost growth measured the brightest shader in the pool, so
    # it also carries the post-breakdown drop at 23:21.9.)
    #   21:42.5 lift   21:46.3 riser   21:51.9 DROP   22:05.0 one-bar break
    #   22:06.9 DROP 2 (the peak, 56 s at full energy)   22:36.9 second half
    #   23:03.2 Last Christmas breakdown   23:21.9 beat back
    #   23:51.4 "I cry..."   24:06.2 thinning into Aperture
    # =====================================================================
    "13 Deck the Halls": {
        "palettes": {"holly":   ["#0B6623", "#FFD700", "#C8102E"],
                     "garland": ["#C8102E", "#0B6623", "#FFF5E1"],
                     "frost":   ["#BDE3FF", "#FFFFFF", "#C8102E"]},
        "palette_family": ["holly", "garland", "frost"],
        "shader_pool": ["frost_growth", "ember_swarm", "crackle_glaze", "lava_lamp"],
        "sections": [(1280.04, "intro"), (1302.54, "build"), (1306.28, "build"),
                     (1311.91, "chorus"), (1325.03, "break"), (1326.89, "chorus"),
                     (1356.91, "chorus"), (1383.16, "break"), (1401.90, "verse"),
                     (1431.44, "verse"), (1446.21, "outro")],
        "section_look": {
            1280.04: {"palette": "garland", "shader": "shader_crackle_glaze"},
            1302.54: {"palette": "garland", "shader": "shader_ember_swarm"},
            1306.28: {"palette": "holly",   "shader": "shader_crackle_glaze"},
            1311.91: {"palette": "holly",   "shader": "shader_ember_swarm"},
            1325.03: {"palette": "garland", "shader": "shader_crackle_glaze"},
            1326.89: {"palette": "holly",   "shader": "shader_ember_swarm"},
            1356.91: {"palette": "garland", "shader": "shader_frost_growth"},
            1383.16: {"palette": "frost",   "shader": "shader_crackle_glaze"},
            1401.90: {"palette": "frost",   "shader": "shader_frost_growth"},
            1431.44: {"palette": "frost",   "shader": "shader_frost_growth"},
            1446.21: {"palette": "frost",   "shader": "shader_lava_lamp"},
        },
        "ambient": [(1385.03, 1401.90)],
        "drops": [1311.91, 1326.89, 1401.90],
        "climax": [1326.89],
    },
    # =====================================================================
    # 14  Harry Styles -- Aperture       24:15.5 - 25:31.2     127.5 BPM
    # The DJ keeps the Last Christmas vocal running over Aperture's build,
    # then the hook "We belong together" lands at 24:36. The last bright
    # look before the finale: WARM WHITE against clear blue with silver --
    # light through a lens -- on optical shaders (mobius spiral, spacels,
    # woven bands). The tempo eases from 127.5 toward Light of Day's 107 at
    # the end (bars stretch 1.87 s -> 2.04 s).
    #   24:15.5 build under "Last Christmas"   24:36.0 "We belong together"
    #   24:58.5 dip   25:06.0 "Let it all out"   25:29.1 down into Light of Day
    # =====================================================================
    "14 Aperture": {
        "palettes": {"aperture": ["#FFF4E0", "#1E90FF", "#C0C8D0"],
                     "lens":     ["#1E90FF", "#FFFFFF", "#7FB2FF"]},
        "palette_family": ["aperture", "lens"],
        "shader_pool": ["mobius_spiral", "spacels", "woven_bands"],
        "sections": [(1455.46, "build"), (1475.97, "chorus"), (1498.48, "verse"),
                     (1505.97, "chorus"), (1529.14, "outro")],
        "section_look": {
            1455.46: {"palette": "lens",     "shader": "shader_woven_bands"},
            1475.97: {"palette": "aperture", "shader": "shader_mobius_spiral"},
            1498.48: {"palette": "lens",     "shader": "shader_spacels"},
            1505.97: {"palette": "aperture", "shader": "shader_mobius_spiral"},
            1529.14: {"palette": "lens",     "shader": "shader_woven_bands"},
        },
        "drops": [1475.97, 1505.97],
        "climax": [1475.97],
    },
    # =====================================================================
    # 15  ODESZA ft. Olafur Arnalds -- Light Of Day   25:31.2 - 28:49.3   107.5
    # The finale. "If I reach the light of day" -- DAWN: gold, peach and
    # sky blue, on the gentlest movers in the set (aurora curtain, falling
    # embers, rain on glass, ripple grid). One last full drop, then the
    # string orchestra: Krists -- "beautiful and flowing... movement but not
    # flashing and then a slow fadeout". From 27:08.8 nothing flashes; the
    # strings swell to 27:43, decay from 27:45.6, and the show fades with
    # them to black by 28:06. The last 37 s of the file are silence.
    #   25:31.2 "If I reach the light of day"   26:04.3 build   26:11.0 bass out
    #   26:15.4 FINAL DROP   26:53.2 heavier second half
    #   27:08.8 strings (no flashing from here)   27:45.6 decay -> fade
    # =====================================================================
    "15 Light Of Day": {
        "palettes": {"daybreak":   ["#FFC857", "#FF9F80", "#87CEEB"],
                     "firstlight": ["#87CEEB", "#FFF1C1", "#FFB38A"]},
        "palette_family": ["daybreak", "firstlight"],
        "shader_pool": ["aurora_curtain", "falling_embers", "rain_on_glass", "ripple_grid"],
        "sections": [(1531.17, "verse"), (1564.33, "build"), (1570.99, "break"),
                     (1576.56, "chorus"), (1613.22, "chorus"), (1628.79, "outro")],
        "section_look": {
            1531.17: {"palette": "firstlight", "shader": "shader_rain_on_glass"},
            1564.33: {"palette": "firstlight", "shader": "shader_ripple_grid"},
            1570.99: {"palette": "daybreak",   "shader": "shader_rain_on_glass"},
            1576.56: {"palette": "daybreak",   "shader": "shader_ripple_grid"},
            1613.22: {"palette": "daybreak",   "shader": "shader_falling_embers"},
            1628.79: {"palette": "daybreak",   "shader": "shader_aurora_curtain"},
        },
        "ambient": [(1628.79, 1729.29)],
        "drops": [1576.56],
        "climax": [1576.56],
        # ONE outro section: a look change at 27:45.6 sat on the fade's first
        # frame and measured as a dip (lit 20% -> 4% -> 9%) as the new beds
        # faded in. The fade follows the strings' decay (27:45.6) and ends a
        # few seconds after their reverb does (~28:05).
        "outro": {"no_flash_from": 1628.79, "fade": [1665.61, 1686.0]},
    },
}
