"""Copy whole model rows from one generated .xsq into another.

    python custom_tools/graft_elements.py <base.xsq> <donor.xsq> <out.xsq> "<model>" ["<model>" ...]

For an approved sequence that needs ONE prop changed: regenerating the whole
song re-rolls passes that depend on anything that moved in the generator since
(Be With You's regenerated final guitar hit gained a one-beat blackout), so
take the approved file and replace only the named models' rows with the
donor's. Palette and EffectDB references are positional, so the donor's are
remapped into the base's tables (appending what is new).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def main() -> None:
    base_p, donor_p, out_p, *names = sys.argv[1:]
    base = ET.parse(base_p); b = base.getroot()
    d = ET.parse(donor_p).getroot()
    bp, bdb = b.find("ColorPalettes"), b.find("EffectDB")
    pal = [(c.text or "") for c in bp]; edb = [(c.text or "") for c in bdb]
    pix = {s: i for i, s in enumerate(pal)}; eix = {s: i for i, s in enumerate(edb)}
    dpal = [(c.text or "") for c in d.find("ColorPalettes")]
    dedb = [(c.text or "") for c in d.find("EffectDB")]

    def gp(s: str) -> int:
        if s not in pix:
            pix[s] = len(pal); pal.append(s); ET.SubElement(bp, "ColorPalette").text = s
        return pix[s]

    def ge(s: str) -> int:
        if s not in eix:
            eix[s] = len(edb); edb.append(s); ET.SubElement(bdb, "Effect").text = s
        return eix[s]

    bee = b.find("ElementEffects")
    for nm in names:
        de = next((e for e in d.find("ElementEffects") if e.get("name") == nm), None)
        if de is None:
            raise SystemExit(f"donor has no element {nm!r}")
        for ef in de.iter("Effect"):
            if ef.get("ref") is not None:
                ef.set("ref", str(ge(dedb[int(ef.get("ref"))])))
            if ef.get("palette") is not None:
                ef.set("palette", str(gp(dpal[int(ef.get("palette"))])))
        old = next((e for e in bee if e.get("name") == nm), None)
        if old is not None:
            bee.insert(list(bee).index(old), de); bee.remove(old)
        else:
            bee.append(de)
        print(f"  grafted {nm}: {sum(1 for _ in de.iter('Effect'))} effects")
    base.write(out_p, encoding="UTF-8", xml_declaration=True)


if __name__ == "__main__":
    main()
