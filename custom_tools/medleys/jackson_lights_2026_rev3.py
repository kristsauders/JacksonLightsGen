"""Jackson Lights 2026 rev3 -- the launch-party medley, 35:45, 21 songs.

Times are MIX time in "Jackson Lights 2026 rev3.wav".

rev3 = rev2 with Every Night's tempo fixed and four songs added before Light Of
Day. Measured against rev2 (chroma tracking): identical to 16:04; Every Night's
rev2 speed-up (109-113 BPM at 16:10-16:30) is gone, so 16:04-16:44 maps through a
ramp and everything after is rev2 +3.594 s; Light Of Day is rev2 +309.7 s with
the strings ~1.6 s quicker. This design is the rev2 design pushed through those
maps (scratchpad maps_r3.py), plus the four new songs.

Reused unchanged (segments.json `reused`): 01-04 (the first mix), 05-09 and
11-16 (rev2's per-song sequences, 11-16 shifted +3.594 s), and Light Of Day
(rev2's intro, then the v13 finale through a composed time map).

The rule for every cut (Krists): "for transitions keep the previous look so it
doesn't give it away" -- each song starts where it is FULLY in.

Shaders stay unique per song; every registered one is owned, so the new songs
get custom shaders (River Current, Sunbeams, Lantern Rise, Heart Lattice), each
with a parameter variant for contrast.
"""

DESIGN: dict[str, dict] = {

    # =====================================================================
    # 05  The 'Last Christmas' vocal blend   6:48.7 - 8:06.1   (new place)
    # The old mix's frost passage (it sat inside Deck the Halls), now its own song
    # leading into Aperture. Frost and mistletoe; Aperture's build under the vocal
    # keeps THIS look until Aperture's drop -- the incoming song is not given away.
    # =====================================================================
    '05 Last Christmas': {'palettes': {'frost': ['#BDE3FF', '#FFFFFF', '#C8102E'],
                  'mistletoe': ['#C8102E', '#FFFFFF', '#0B6623']},
     'palette_family': ['frost', 'mistletoe'],
     'shader_pool': ['frost_growth', 'crackle_glaze'],
     'sections': [(408.68, 'verse'),
                  (424.17, 'verse'),
                  (439.65, 'chorus'),
                  (455.14, 'break'),
                  (466.75, 'build')],
     'section_look': {408.68: {'palette': 'frost', 'shader': 'shader_frost_growth'},
                      424.17: {'palette': 'mistletoe', 'shader': 'shader_crackle_glaze'},
                      439.65: {'palette': 'frost', 'shader': 'shader_frost_growth'},
                      455.14: {'palette': 'mistletoe', 'shader': 'shader_crackle_glaze'},
                      466.75: {'palette': 'frost', 'shader': 'shader_frost_growth'}},
     'drops': []},

    # =====================================================================
    # 06  Harry Styles - Aperture        8:06.1 - 10:17.5    extended edit
    # Same colours/shaders as before. Drops: 'We belong together' 8:06.1, 'Let it
    # all out' 8:37.1, the added third chorus 9:15.7 (climax); fades to 10:11.7.
    # =====================================================================
    '06 Aperture': {'palettes': {'aperture': ['#FFF4E0', '#1E90FF', '#C0C8D0'],
                  'lens': ['#1E90FF', '#FFFFFF', '#7FB2FF']},
     'palette_family': ['aperture', 'lens'],
     'shader_pool': ['aperture_field', 'spacels_slow', 'woven_bands'],
     # rev3 notes: the purple Mobius Spiral read too like an earlier song's
     # spinner -> Aperture Field (written for this song: a honeycomb of camera
     # irises opening in waves); Spacels' stars were "moving way too fast" ->
     # the same shader at 0.35x clock. Woven Bands from 9:00 kept.
     'extra_shaders': ['Aperture Field', {'file': 'Spacels', 'key': 'spacels_slow', 'speed': 0.35}],
     'sections': [(486.10, 'chorus'),
                  (509.33, 'verse'),
                  (517.07, 'chorus'),
                  (540.30, 'break'),
                  (553.85, 'build'),
                  (555.74, 'chorus'),
                  (611.74, 'outro')],
     'section_look': {486.10: {'palette': 'aperture', 'shader': 'shader_aperture_field'},
                      509.33: {'palette': 'lens', 'shader': 'shader_spacels_slow'},
                      517.07: {'palette': 'aperture', 'shader': 'shader_aperture_field'},
                      540.30: {'palette': 'lens', 'shader': 'shader_woven_bands'},
                      553.85: {'palette': 'lens', 'shader': 'shader_woven_bands'},
                      555.74: {'palette': 'aperture', 'shader': 'shader_woven_bands'},
                      611.74: {'palette': 'lens', 'shader': 'shader_woven_bands'}},
     'drops': [486.10, 517.07, 555.74],
     'climax': [555.74]},

    # =====================================================================
    # 07  Gigi D'Agostino - Star         10:17.5 - 11:05.4   new bridge
    # 1999 Italo-dance from L'Amour Toujours: dreamy, starry. Indigo, cyan, white.
    # =====================================================================
    '07 Star': {'palettes': {'stardust': ['#2A1B7A', '#FFFFFF', '#3BD6FF'],
                  'nova': ['#3BD6FF', '#B388FF', '#FFFFFF']},
     'palette_family': ['stardust', 'nova'],
     'shader_pool': ['caustic_pool', 'ribbon_braid', 'wind_wheat'],
     'sections': [(617.50, 'verse'), (630.83, 'chorus'), (647.97, 'chorus'), (661.30, 'outro')],
     'section_look': {617.50: {'palette': 'nova', 'shader': 'shader_ribbon_braid'},
                      630.83: {'palette': 'stardust', 'shader': 'shader_caustic_pool'},
                      647.97: {'palette': 'nova', 'shader': 'shader_wind_wheat'},
                      661.30: {'palette': 'stardust', 'shader': 'shader_ribbon_braid'}},
     'drops': [630.83]},

    # =====================================================================
    # 08  Bad Bunny - NUEVAYoL   11:05.4 - 12:12.7   identical audio, old design -44.5 s
    # =====================================================================
    '08 NUEVAYoL': {'palettes': {'boricua': ['#E4002B', '#0050F0', '#FFFFFF'],
                  'nuevayol': ['#FFC72C', '#E4002B', '#0050F0']},
     'palette_family': ['boricua', 'nuevayol'],
     'shader_pool': ['circuit_trace', 'herringbone', 'fracture'],
     'sections': [(665.40, 'intro'), (679.59, 'chorus'), (697.39, 'verse'), (712.64, 'verse')],
     'section_look': {665.40: {'palette': 'nuevayol', 'shader': 'shader_herringbone'},
                      679.59: {'palette': 'boricua', 'shader': 'shader_fracture'},
                      697.39: {'palette': 'nuevayol', 'shader': 'shader_circuit_trace'},
                      712.64: {'palette': 'boricua', 'shader': 'shader_fracture'}},
     'drops': [679.59],
     'climax': [679.59],
     'pulse_mode': 'groove'},

    # =====================================================================
    # 09  CloZee & LSDREAM - ENIGMA   12:12.7 - 14:21.3   identical, old design -47.6 s
    # =====================================================================
    '09 ENIGMA': {'palettes': {'enigma': ['#7CFF00', '#5B0E91', '#00FFC6'],
                  'ritual': ['#5B0E91', '#B6FF3B', '#FF7A00']},
     'palette_family': ['enigma', 'ritual'],
     'shader_pool': ['murmuration', 'ink_bloom', 'interference', 'hex_vortex'],
     'sections': [(732.66, 'intro'),
                  (759.80, 'chorus'),
                  (777.16, 'break'),
                  (779.60, 'chorus'),
                  (800.06, 'break'),
                  (814.94, 'build'),
                  (819.20, 'chorus'),
                  (836.55, 'verse'),
                  (856.94, 'outro')],
     'section_look': {732.66: {'palette': 'ritual', 'shader': 'shader_murmuration'},
                      759.80: {'palette': 'enigma', 'shader': 'shader_hex_vortex'},
                      777.16: {'palette': 'ritual', 'shader': 'shader_interference'},
                      779.60: {'palette': 'enigma', 'shader': 'shader_ink_bloom'},
                      800.06: {'palette': 'ritual', 'shader': 'shader_murmuration'},
                      814.94: {'palette': 'ritual', 'shader': 'shader_interference'},
                      819.20: {'palette': 'enigma', 'shader': 'shader_hex_vortex'},
                      836.55: {'palette': 'ritual', 'shader': 'shader_ink_bloom'},
                      856.94: {'palette': 'enigma', 'shader': 'shader_interference'}},
     'ambient': [(732.66, 757.99)],
     'drops': [759.80, 779.60, 819.20],
     'climax': [819.20]},

    # =====================================================================
    # 10  Spectacle - Every Night   14:21.3 - 17:52.8   re-edited (offset drifts -45 -> -50 s):
    # old design mapped through the measured alignment, then refined to rev2's beats
    # =====================================================================
    '10 Every Night': {'palettes': {'nocturne': ['#2B1B8F', '#8A2BE2', '#00D4FF'],
                  'afterglow': ['#00D4FF', '#6A00FF', '#FF5EDB']},
     'palette_family': ['nocturne', 'afterglow'],
     'shader_pool': ['drift_cells', 'contour_drift', 'tide_lines', 'pulse_rings'],
     'sections': [(858.81, 'verse'),
                  (893.30, 'chorus'),
                  (917.77, 'break'),
                  (927.59, 'break'),
                  (939.85, 'build'),
                  (952.69, 'verse'),
                  (974.12, 'break'),
                  (983.92, 'chorus'),
                  (1003.51, 'verse'),
                  (1023.07, 'chorus'),
                  (1059.78, 'outro')],
     'section_look': {858.81: {'palette': 'nocturne', 'shader': 'shader_pulse_rings'},
                      893.30: {'palette': 'afterglow', 'shader': 'shader_drift_cells'},
                      917.77: {'palette': 'nocturne', 'shader': 'shader_tide_lines'},
                      927.59: {'palette': 'nocturne', 'shader': 'shader_contour_drift'},
                      939.85: {'palette': 'afterglow', 'shader': 'shader_tide_lines'},
                      952.69: {'palette': 'nocturne', 'shader': 'shader_pulse_rings'},
                      974.12: {'palette': 'nocturne', 'shader': 'shader_contour_drift'},
                      983.92: {'palette': 'afterglow', 'shader': 'shader_drift_cells'},
                      1003.51: {'palette': 'nocturne', 'shader': 'shader_tide_lines'},
                      1023.07: {'palette': 'afterglow', 'shader': 'shader_drift_cells'},
                      1059.78: {'palette': 'nocturne', 'shader': 'shader_contour_drift'}},
     'ambient': [(927.59, 939.85)],
     'drops': [983.92],
     'climax': [1023.07]},

    # =====================================================================
    # 11  Happy Holidays (Beef Wellington)   17:52.8 - 19:26.7   old design mapped (~-49.7 s)
    # =====================================================================
    '11 Happy Holidays': {'palettes': {'fireside': ['#A4123F', '#D4A017', '#FFF3D6'],
                  'tinsel': ['#D4A017', '#FFF3D6', '#2E6B3F']},
     'palette_family': ['fireside', 'tinsel'],
     'shader_pool': ['velvet_nap', 'oil_film', 'marble_vein'],
     'sections': [(1072.77, 'intro'),
                  (1081.30, 'build'),
                  (1083.28, 'chorus'),
                  (1103.56, 'verse'),
                  (1125.18, 'chorus'),
                  (1143.54, 'break'),
                  (1146.18, 'chorus'),
                  (1161.89, 'outro')],
     'section_look': {1072.77: {'palette': 'tinsel', 'shader': 'shader_velvet_nap'},
                      1081.30: {'palette': 'tinsel', 'shader': 'shader_oil_film'},
                      1083.28: {'palette': 'fireside', 'shader': 'shader_oil_film'},
                      1103.56: {'palette': 'tinsel', 'shader': 'shader_marble_vein'},
                      1125.18: {'palette': 'fireside', 'shader': 'shader_oil_film'},
                      1143.54: {'palette': 'tinsel', 'shader': 'shader_velvet_nap'},
                      1146.18: {'palette': 'fireside', 'shader': 'shader_oil_film'},
                      1161.89: {'palette': 'tinsel', 'shader': 'shader_velvet_nap'}},
     'ambient': [(1072.77, 1081.30)],
     'drops': [1083.28],
     'climax': [1146.18]},

    # =====================================================================
    # 12  Santa Claus Is Coming to Town   19:26.7 - 20:28.3   old design -49.9 s;
    # its second drop now opens Deck the Halls (song 15)
    # =====================================================================
    '12 Santa Claus Is Coming to Town': {'palettes': {'candycane': ['#FF0A3C', '#FFFFFF', '#FF7A9A'],
                  'peppermint': ['#FFFFFF', '#FF0A3C', '#00B37A']},
     'palette_family': ['candycane', 'peppermint'],
     # rev3 notes: "very rhythmic right away even though there's not much rhythm
     # yet... the main rhythm doesn't hit until the drop" -- drums < 25% of the
     # song's level until 19:42, gentle kick only to the drop at 19:57.3.
     'ambient': [(1166.68, 1197.30)],
     'shader_pool': ['kaleido_bloom', 'target_circle', 'innerdimensionalmatrix'],
     'sections': [(1166.68, 'verse'),
                  (1181.96, 'build'),
                  (1193.47, 'break'),
                  (1197.30, 'chorus'),
                  (1212.67, 'chorus'),
                  (1226.09, 'break')],
     'section_look': {1166.68: {'palette': 'peppermint', 'shader': 'shader_target_circle'},
                      1181.96: {'palette': 'peppermint', 'shader': 'shader_kaleido_bloom'},
                      1193.47: {'palette': 'candycane', 'shader': 'shader_target_circle'},
                      1197.30: {'palette': 'candycane',
                                'shader': 'shader_innerdimensionalmatrix'},
                      1212.67: {'palette': 'peppermint', 'shader': 'shader_kaleido_bloom'},
                      1226.09: {'palette': 'candycane', 'shader': 'shader_target_circle'}},
     'drops': [1197.30],
     'climax': [1197.30]},

    # =====================================================================
    # 13  Zara Larsson & MNEK - Never Forget You (guaracha edit)  20:28.3 - 22:41.9
    # ~145 BPM Latin-house: coral, turquoise, sunshine and neon. Drops after a
    # one-bar silence at 21:18.85 and 22:15.4 (the peak).
    # =====================================================================
    '13 Never Forget You': {'palettes': {'guaracha': ['#FF5A36', '#00D1C1', '#FFD23F'],
                  'neon': ['#FF2BD6', '#00D1C1', '#FFFFFF']},
     'palette_family': ['guaracha', 'neon'],
     # rev3 notes: 20:28-21:22 has no drums (32 bars under 25% of the song's
     # drum level) but the props flashed on chords/hats -> ambient to the drop;
     # the 144 BPM guaracha kicks flash only on the beat grid (snare untouched).
     'ambient': [(1228.32, 1282.44)],
     'voice_on_beat': True,
     'shader_pool': ['iso_blocks', 'liquid_chrome', 'chromatophore'],
     'sections': [(1228.32, 'intro'),
                  (1255.96, 'verse'),
                  (1282.44, 'chorus'),
                  (1308.94, 'break'),
                  (1311.42, 'verse'),
                  (1335.40, 'chorus')],
     'section_look': {1228.32: {'palette': 'neon', 'shader': 'shader_chromatophore'},
                      1255.96: {'palette': 'guaracha', 'shader': 'shader_liquid_chrome'},
                      1282.44: {'palette': 'guaracha', 'shader': 'shader_iso_blocks'},
                      1308.94: {'palette': 'neon', 'shader': 'shader_liquid_chrome'},
                      1311.42: {'palette': 'neon', 'shader': 'shader_liquid_chrome'},
                      1335.40: {'palette': 'neon', 'shader': 'shader_iso_blocks'}},
     'drops': [1282.44, 1335.40],
     'climax': [1335.40],
     # was 'groove' (accents on the kick pattern) -- 31% of its flashes landed
     # off the beat; Krists wants the fast guaracha kicks shown ON the beat.
     'pulse_mode': 'quarter'},

    # =====================================================================
    # 14  The Ronettes - Sleigh Ride (dance edit)   22:41.9 - 23:50.1
    # Spector's 1963 jingles and hooves over a beat: sweater-knit shaders (argyle,
    # tartan, cable knit) in sleigh red/gold and evergreen. Snare roll -> drop 23:08.1.
    # =====================================================================
    # cut time: the tracker hears 182 BPM, the felt pulse is 91
    '14 Sleigh Ride': {'half_time': True,
                       'palettes': {'sleigh': ['#E3001B', '#FFFFFF', '#FFC82E'],
                  'evergreen': ['#0E8A3E', '#FFFFFF', '#E3001B']},
     'palette_family': ['sleigh', 'evergreen'],
     'shader_pool': ['argyle_march', 'tartan_shift', 'cable_knit'],
     'sections': [(1361.86, 'intro'),
                  (1375.01, 'build'),
                  (1388.10, 'chorus'),
                  (1409.08, 'chorus')],
     'section_look': {1361.86: {'palette': 'evergreen', 'shader': 'shader_cable_knit'},
                      1375.01: {'palette': 'sleigh', 'shader': 'shader_tartan_shift'},
                      1388.10: {'palette': 'sleigh', 'shader': 'shader_argyle_march'},
                      1409.08: {'palette': 'evergreen', 'shader': 'shader_tartan_shift'}},
     'ambient': [(1361.86, 1375.01)],
     'drops': [1388.10],
     'climax': [1388.10]},

    # =====================================================================
    # 15  Deck the Halls (J3, eXecutive Decision, Kevin Aleksander)  23:50.1 - 25:56.6
    # Opens on the old Santa Claus second drop (a 5.6 s reprise), then Deck's intro
    # 23:55.7, drops 24:27.6 and 24:42.6 (climax), fades from 25:41. The Last
    # Christmas passage that followed it in the old mix is now song 05.
    # =====================================================================
    '15 Deck the Halls': {'palettes': {'holly': ['#0B6623', '#FFD700', '#C8102E'],
                  'garland': ['#C8102E', '#0B6623', '#FFF5E1']},
     'palette_family': ['holly', 'garland'],
     # rev3 notes: "the songs starting with deck the halls and after are all very
     # dancey and it's the peak of the show" -> the WORKS (verticals level-metering,
     # arches leaping, floods blooming, spinners reversing, canvases running back
     # and forth on the beat) in this song's colours, at 0.9 so Light Of Day's
     # works stays the biggest moment of the show.
     # Intro: "just warming up with no drums yet, so let's hold off the rhythm
     # until the big drop around 24:27" (drums < 25% for 14 bars).
     'ambient': [(1435.72, 1467.58)],
     'scenes': [{'kind': 'works', 'window': [1467.58, 1480.71], 'level': 0.9,
                 'colors': ['#FFD700', '#E0102E', '#FFF5E1', '#00B34A']},
                {'kind': 'works', 'window': [1482.56, 1544.64], 'level': 0.9,
                 'colors': ['#FFD700', '#E0102E', '#FFF5E1', '#00B34A']}],
     # v1 measured the drops/choruses at 0.12-0.26 -- no brighter than the intro:
     # ember_swarm is sparse points. Garland Twist (written for this song) fills
     # the canvas with braided ribbons for the hits; ember_swarm keeps the builds.
     'shader_pool': ['garland_twist', 'ember_swarm', 'lava_lamp'],
     'extra_shaders': ['Garland Twist'],
     'sections': [(1430.08, 'chorus'),
                  (1435.72, 'intro'),
                  (1458.22, 'build'),
                  (1461.96, 'build'),
                  (1467.58, 'chorus'),
                  (1480.71, 'break'),
                  (1482.56, 'chorus'),
                  (1512.60, 'chorus'),
                  (1544.64, 'outro')],
     'section_look': {1430.08: {'palette': 'holly', 'shader': 'shader_garland_twist'},
                      1435.72: {'palette': 'garland', 'shader': 'shader_lava_lamp'},
                      1458.22: {'palette': 'garland', 'shader': 'shader_ember_swarm'},
                      1461.96: {'palette': 'holly', 'shader': 'shader_lava_lamp'},
                      1467.58: {'palette': 'holly', 'shader': 'shader_garland_twist'},
                      1480.71: {'palette': 'garland', 'shader': 'shader_lava_lamp'},
                      1482.56: {'palette': 'holly', 'shader': 'shader_garland_twist'},
                      1512.60: {'palette': 'garland', 'shader': 'shader_garland_twist'},
                      1544.64: {'palette': 'garland', 'shader': 'shader_lava_lamp'}},
     'drops': [1430.08, 1467.58, 1482.56],
     'climax': [1482.56]},

    # =====================================================================
    # 16  Bonobo - Shokoufeh ('blossom')   25:56.6 - 27:01.7
    # From Distance In Static (2026). A custom Blossom Drift shader: petals drifting
    # and turning over a slow wash, in pinks over teal / dusk violet.
    # =====================================================================
    '16 Shokoufeh': {'palettes': {'blossom': ['#FF7FA8', '#FFD6E7', '#2EC4B6'],
                  'dusk': ['#FF7FA8', '#FFC27A', '#6B2D9C']},
     'palette_family': ['blossom', 'dusk'],
     # v1 measured the last chorus (0.14) dimmer than the verses (0.18-0.21) at
     # full energy: the choruses get the same petals, denser and bigger, on a
     # brighter wash ("blossom_bloom"). The 4 s break before the last chorus is
     # under the 4 s min dwell, so its span merges INTO the chorus and lends it
     # its shader: the break carries the chorus look (its curve keeps it dark).
     'shader_pool': ['blossom_drift', 'blossom_bloom'],
     'extra_shaders': ['Blossom Drift',
                       {'file': 'Blossom Drift', 'key': 'blossom_bloom',
                        'overrides': {'density': 0.85, 'size': 0.55, 'wash': 0.7}}],
     'sections': [(1556.58, 'intro'),
                  (1560.64, 'verse'),
                  (1572.84, 'chorus'),
                  (1589.12, 'verse'),
                  (1593.18, 'chorus'),
                  (1605.39, 'break'),
                  (1609.46, 'chorus')],
     'section_look': {1556.58: {'palette': 'dusk', 'shader': 'shader_blossom_drift'},
                      1560.64: {'palette': 'blossom', 'shader': 'shader_blossom_drift'},
                      1572.84: {'palette': 'dusk', 'shader': 'shader_blossom_bloom'},
                      1589.12: {'palette': 'blossom', 'shader': 'shader_blossom_drift'},
                      1593.18: {'palette': 'dusk', 'shader': 'shader_blossom_bloom'},
                      1605.39: {'palette': 'dusk', 'shader': 'shader_blossom_bloom'},
                      1609.46: {'palette': 'blossom', 'shader': 'shader_blossom_bloom'}},
     'drops': [1572.84]},

    # =====================================================================
    # 17  Gregory Porter - Liquid Spirit (Claptone remix)   27:01.7 - 28:11.2   120 BPM
    # Deep house under a gospel-jazz voice: "un-dam the rivers". Custom River Current
    # shader (meandering channels, gold current streaks); a fuller variant for the
    # second half. Shokoufeh stops dead at 27:01.7 -> 4 quiet bars -> the beat.
    #   27:01.7 quiet intro   27:09.7 the beat lands   27:41.5 second half ("Now.")
    #   27:59.3 fades out under Movin' To The Sun's intro
    # =====================================================================
    '17 Liquid Spirit': {'palettes': {'river': ['#00A6D6', '#FFC94A', '#0B1E5B'],
                  'spirit': ['#2EC4B6', '#FFFFFF', '#3A0CA3']},
     'palette_family': ['river', 'spirit'],
     # rev3 notes: "the songs starting with deck the halls and after are all very
     # dancey and it's the peak of the show" -> the WORKS (verticals level-metering,
     # arches leaping, floods blooming, spinners reversing, canvases running back
     # and forth on the beat) in this song's colours (second half), at 0.9 so Light Of Day's
     # works stays the biggest moment of the show.
     'scenes': [{'kind': 'works', 'window': [1661.45, 1679.30], 'level': 0.9,
                 'colors': ['#FFC94A', '#00A6D6', '#FFFFFF', '#3A0CA3']}],
     'shader_pool': ['river_current', 'river_flood'],
     'extra_shaders': ['River Current',
                       {'file': 'River Current', 'key': 'river_flood',
                        'overrides': {'bands': 4.0, 'meander': 0.9, 'streaks': 0.9}}],
     'sections': [(1621.67, 'intro'),
                  (1629.72, 'chorus'),
                  (1661.45, 'chorus'),
                  (1679.30, 'outro')],
     'section_look': {1621.67: {'palette': 'spirit', 'shader': 'shader_river_current'},
                      1629.72: {'palette': 'river', 'shader': 'shader_river_current'},
                      1661.45: {'palette': 'spirit', 'shader': 'shader_river_flood'},
                      1679.30: {'palette': 'river', 'shader': 'shader_river_current'}},
     'ambient': [(1621.67, 1629.72)],
     'drops': [1629.72]},

    # =====================================================================
    # 18  HUGEL, Imael Angel & Ultra Nate - Movin' To The Sun   28:11.2 - 29:14.8   125 BPM
    # Sunny Afro/Latin house. Custom Sunbeams shader (rays from above the house,
    # swinging); a blazing variant for the drop. Stops DEAD at 29:14.8.
    #   28:11.2 vocal intro over hats   28:28.7 bass in   28:40.3 bass out (riser)
    #   28:44.1 the drop   29:14.8 dead stop
    # =====================================================================
    "18 Movin' To The Sun": {'palettes': {'sunrise': ['#FFD166', '#FF7B00', '#9D0208'],
                  'tropic': ['#FFF3B0', '#FF5D8F', '#3A0CA3']},
     'palette_family': ['sunrise', 'tropic'],
     # rev3 notes: "more rhythmic movement in the intro which is driving but
     # doesn't have drums, and when the drums start it almost feels like there's
     # less" -> intro ambient (hats, no kick) to the bass at 28:28.7; the drop
     # gets the works.
     # rev3 notes: "the songs starting with deck the halls and after are all very
     # dancey and it's the peak of the show" -> the WORKS (verticals level-metering,
     # arches leaping, floods blooming, spinners reversing, canvases running back
     # and forth on the beat) in this song's colours, at 0.9 so Light Of Day's
     # works stays the biggest moment of the show.
     'ambient': [(1691.21, 1708.73)],
     'scenes': [{'kind': 'works', 'window': [1724.08, 1754.81], 'level': 0.9,
                 'colors': ['#FFD166', '#FF5D8F', '#FFF3B0', '#FF7B00']}],
     'shader_pool': ['sunbeams', 'sunbeams_blaze'],
     'extra_shaders': ['Sunbeams',
                       {'file': 'Sunbeams', 'key': 'sunbeams_blaze',
                        'overrides': {'rays': 14.0, 'spread': 0.85, 'motes': 0.2}}],
     'sections': [(1691.21, 'intro'),
                  (1708.73, 'verse'),
                  (1720.25, 'build'),
                  (1724.08, 'chorus')],
     'section_look': {1691.21: {'palette': 'tropic', 'shader': 'shader_sunbeams'},
                      1708.73: {'palette': 'sunrise', 'shader': 'shader_sunbeams'},
                      1720.25: {'palette': 'tropic', 'shader': 'shader_sunbeams'},
                      1724.08: {'palette': 'sunrise', 'shader': 'shader_sunbeams_blaze'}},
     'drops': [1724.08]},

    # =====================================================================
    # 19  Elderbrook & George FitzGerald - Glad I Found You   29:14.8 - 30:18.7   120 BPM
    # A shorter edit than the first mix's, so new visuals -- its old sunset/lagoon
    # palettes, on a custom Lantern Rise shader (warm lanterns rising into dusk).
    #   29:14.8 near-silence after the stop   29:29.2 the melody (bass in)   29:42.3 one bar of
    #   silence   29:44.3 synth back   29:46.2 the bass lands (beat 3 of the tracker's bar --
    #   its phase is two beats off through this blend)   30:15 hands over to What Love Can Do (128 BPM)
    # =====================================================================
    '19 Glad I Found You': {'palettes': {'sunset': ['#FF6F59', '#FFB38A', '#13A89E'],
                  'lagoon': ['#13A89E', '#7FDBDA', '#FF8C69']},
     'palette_family': ['sunset', 'lagoon'],
     # rev3 notes: "the intro has lots of visual rhythm even though it's pretty
     # ambient... save the rhythm for the drop and chorus" -> ambient (drums < 25%
     # for 15 bars) right up to the bass at 29:46.2, then the works.
     # rev3 notes: "the songs starting with deck the halls and after are all very
     # dancey and it's the peak of the show" -> the WORKS (verticals level-metering,
     # arches leaping, floods blooming, spinners reversing, canvases running back
     # and forth on the beat) in this song's colours, at 0.9 so Light Of Day's
     # works stays the biggest moment of the show.
     'scenes': [{'kind': 'works', 'window': [1786.24, 1818.74], 'level': 0.9,
                 'colors': ['#FF6F59', '#13A89E', '#FFB38A', '#7FDBDA']}],
     'shader_pool': ['lantern_rise', 'lantern_swarm'],
     'extra_shaders': ['Lantern Rise',
                       {'file': 'Lantern Rise', 'key': 'lantern_swarm',
                        'overrides': {'density': 0.9, 'size': 0.3, 'sky': 0.7}}],
     'sections': [(1754.81, 'intro'),
                  (1769.24, 'verse'),
                  (1782.25, 'break'),
                  (1784.25, 'build'),
                  (1786.24, 'chorus')],
     'section_look': {1754.81: {'palette': 'lagoon', 'shader': 'shader_lantern_rise'},
                      1769.24: {'palette': 'sunset', 'shader': 'shader_lantern_rise'},
                      1782.25: {'palette': 'lagoon', 'shader': 'shader_lantern_rise'},
                      1784.25: {'palette': 'lagoon', 'shader': 'shader_lantern_rise'},
                      1786.24: {'palette': 'sunset', 'shader': 'shader_lantern_swarm'}},
     'ambient': [(1754.81, 1786.24)],
     'drops': [1786.24]},

    # =====================================================================
    # 20  BONNIE X CLYDE - What Love Can Do   30:18.7 - 32:21.2   128 BPM
    # The last big drop before the finale. Custom Heart Lattice shader (diamond
    # cells lighting in rings that spread like a heartbeat); full and soft
    # variants. Both drops come after ONE silent bar -- the blackout is the moment.
    #   30:18.7 vocal   30:34.3 silent bar   30:36.1 DROP   30:50.2 bass out 2 beats -> 30:51.1 groove
    #   31:19.3 breakdown   31:30.5 build   31:49.3 silent bar   31:51.2 DROP 2
    #   32:21.2 hard stop into Light Of Day
    # =====================================================================
    '20 What Love Can Do': {'palettes': {'love': ['#FF1053', '#FFC2D4', '#6A00F4'],
                  'electric': ['#F72585', '#4CC9F0', '#7209B7']},
     'palette_family': ['love', 'electric'],
     # rev3 notes: Heart Lattice was "moving too fast, looks like it's strobing"
     # -> the shader itself is slower with soft cells; the 31:30 build keeps its
     # rhythm ("I love the current visuals... keep the rhythm there").
     # rev3 notes: "the songs starting with deck the halls and after are all very
     # dancey and it's the peak of the show" -> the WORKS (verticals level-metering,
     # arches leaping, floods blooming, spinners reversing, canvases running back
     # and forth on the beat) in this song's colours, at 0.9 so Light Of Day's
     # works stays the biggest moment of the show.
     'scenes': [{'kind': 'works', 'window': [1836.14, 1851.14], 'level': 0.9,
                 'colors': ['#FF1053', '#4CC9F0', '#FFC2D4', '#7209B7']},
                {'kind': 'works', 'window': [1911.15, 1941.16], 'level': 0.9,
                 'colors': ['#FF1053', '#4CC9F0', '#FFC2D4', '#7209B7']}],
     'shader_pool': ['heart_lattice', 'heart_full', 'heart_soft'],
     'extra_shaders': ['Heart Lattice',
                       {'file': 'Heart Lattice', 'key': 'heart_full',
                        'overrides': {'fill': 0.85, 'scale': 8.0, 'lines': 0.7}},
                       {'file': 'Heart Lattice', 'key': 'heart_soft',
                        'overrides': {'fill': 0.45, 'scale': 4.0, 'lines': 0.3}}],
     'sections': [(1818.74, 'verse'),
                  (1834.25, 'build'),
                  (1836.14, 'chorus'),
                  (1851.14, 'verse'),
                  (1879.27, 'break'),
                  (1890.53, 'build'),
                  (1911.15, 'chorus')],
     'section_look': {1818.74: {'palette': 'electric', 'shader': 'shader_heart_soft'},
                      1834.25: {'palette': 'electric', 'shader': 'shader_heart_soft'},
                      1836.14: {'palette': 'love', 'shader': 'shader_heart_full'},
                      1851.14: {'palette': 'electric', 'shader': 'shader_heart_lattice'},
                      1879.27: {'palette': 'love', 'shader': 'shader_heart_soft'},
                      1890.53: {'palette': 'electric', 'shader': 'shader_heart_lattice'},
                      1911.15: {'palette': 'love', 'shader': 'shader_heart_full'}},
     'drops': [1836.14, 1911.15],
     'climax': [1911.15]},

    # =====================================================================
    # 21  ODESZA - Light Of Day   32:21.2 - 35:45.2   the v13 design, mapped from the FIRST
    # mix directly (old + 420.90 s through the final chorus; the strings are slower,
    # +421.0 -> +423.8 s, measured by chroma at 1 s steps). Regenerated for rev3 (the
    # EFL flakes spin with the others; Krists: "smooth for that section, until the
    # drums come in at 33:17", so 32:21-33:17 is ambient). The strings stop at
    # 34:46.70 (-17 -> -27 dB in 0.3 s; v13's 27:45.0 was the same moment) -- the
    # stillness is pinned there, the fade starts 0.61 s later as in v13.
    # =====================================================================
    '21 Light Of Day': {'palettes': {'daybreak': ['#FFC857', '#FF9F80', '#87CEEB'],
                   'firstlight': ['#87CEEB', '#FFF1C1', '#FFB38A']},
      'palette_family': ['daybreak', 'firstlight'],
      'shader_pool': ['aurora_curtain', 'falling_embers', 'rain_on_glass', 'ripple_grid'],
      'sections': [(1941.16, 'intro'),
                   (1952.07, 'verse'),
                   (1985.23, 'build'),
                   (1991.89, 'break'),
                   (1997.46, 'chorus'),
                   (2034.12, 'chorus'),
                   (2049.69, 'outro')],
      'section_look': {1941.16: {'palette': 'firstlight', 'shader': 'shader_rain_on_glass'},
                       1952.07: {'palette': 'firstlight', 'shader': 'shader_rain_on_glass'},
                       1985.23: {'palette': 'firstlight', 'shader': 'shader_ripple_grid'},
                       1991.89: {'palette': 'daybreak', 'shader': 'shader_rain_on_glass'},
                       1997.46: {'palette': 'daybreak', 'shader': 'shader_ripple_grid'},
                       2034.12: {'palette': 'daybreak', 'shader': 'shader_ripple_grid'},
                       2049.69: {'palette': 'daybreak', 'shader': 'shader_aurora_curtain'}},
      'ambient': [(1941.16, 1997.46), (2049.69, 2145.22)],
      'drops': [1997.46],
      'climax': [1997.46],
      'outro': {'no_flash_from': 2049.69, 'fade': [2087.31, 2109.06]},
      'scenes': [{'kind': 'works', 'window': [1997.46, 2049.69]},
                 {'kind': 'horizon',
                  'window': [2049.69, 2145.22],
                  'peak': 2084.85,
                  'fade': [2087.31, 2109.06],
                  'still_from': 2086.7,
                  'strokes': [2055.21,
                              2056.33,
                              2056.97,
                              2057.91,
                              2059.04,
                              2059.7,
                              2060.31,
                              2060.87,
                              2061.43,
                              2062.06,
                              2062.67,
                              2063.51,
                              2064.31,
                              2064.84,
                              2065.42,
                              2065.99,
                              2066.69,
                              2067.42,
                              2067.91,
                              2068.42,
                              2069.18,
                              2069.68,
                              2070.39,
                              2070.98,
                              2071.49,
                              2072.46,
                              2072.95,
                              2073.75,
                              2074.44,
                              2075.19,
                              2075.79,
                              2076.53,
                              2077.43,
                              2078.14,
                              2079.26,
                              2079.82,
                              2080.44,
                              2081.47,
                              2082.47,
                              2083.52,
                              2084.09,
                              2084.67,
                              2085.17,
                              2085.88]}]}}
