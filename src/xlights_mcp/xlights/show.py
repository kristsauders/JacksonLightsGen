"""Parse xLights show folder configuration files."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from xlights_mcp.xlights.models import (
    Controller,
    LightModel,
    ModelGroup,
    ShowConfig,
    SubModel,
)

logger = logging.getLogger(__name__)


def load_show_config(show_path: Path) -> ShowConfig:
    """Load complete show configuration from an xLights show folder.

    Reads xlights_networks.xml and xlights_rgbeffects.xml to build
    a complete picture of the controllers, models, and groups.
    """
    show_name = show_path.name
    controllers = load_show_controllers(show_path)
    models = load_show_models(show_path)
    groups = load_model_groups(show_path)
    total_channels = sum(c.max_channels for c in controllers)

    return ShowConfig(
        show_path=str(show_path),
        show_name=show_name,
        controllers=controllers,
        models=models,
        model_groups=groups,
        total_channels=total_channels,
    )


def load_show_controllers(show_path: Path) -> list[Controller]:
    """Parse xlights_networks.xml to extract controller configuration."""
    networks_file = show_path / "xlights_networks.xml"
    if not networks_file.exists():
        logger.warning(f"Networks file not found: {networks_file}")
        return []

    tree = ET.parse(networks_file)
    root = tree.getroot()
    controllers = []

    for ctrl_elem in root.findall("Controller"):
        max_channels = 0
        for network in ctrl_elem.findall("network"):
            ch = network.get("MaxChannels", "0")
            try:
                max_channels += int(ch)
            except ValueError:
                pass

        controller = Controller(
            id=ctrl_elem.get("Id", ""),
            name=ctrl_elem.get("Name", ""),
            description=ctrl_elem.get("Description", ""),
            controller_type=ctrl_elem.get("Type", ""),
            vendor=ctrl_elem.get("Vendor", ""),
            model=ctrl_elem.get("Model", ""),
            ip=ctrl_elem.get("IP", ""),
            protocol=ctrl_elem.get("Protocol", ""),
            max_channels=max_channels,
            active_state=ctrl_elem.get("ActiveState", ""),
        )
        controllers.append(controller)

    logger.info(f"Loaded {len(controllers)} controllers from {networks_file}")
    return controllers


def load_show_models(show_path: Path) -> list[LightModel]:
    """Parse xlights_rgbeffects.xml to extract model definitions."""
    effects_file = show_path / "xlights_rgbeffects.xml"
    if not effects_file.exists():
        logger.warning(f"RGB effects file not found: {effects_file}")
        return []

    tree = ET.parse(effects_file)
    root = tree.getroot()
    models_elem = root.find("models")
    if models_elem is None:
        return []

    models = []
    for m in models_elem:
        if m.tag == "model":
            # Collect submodels (child elements)
            submodels = []
            face_definitions = []
            for child in m:
                if child.tag == "faceInfo":
                    face_name = child.get("Name", "")
                    if face_name:
                        face_definitions.append(face_name)
                    continue
                if child.tag == "subModel":
                    sub_name = child.get("name", "")
                    if sub_name:
                        submodels.append(SubModel(name=sub_name, parent=m.get("name", "")))
                    continue
                if child.tag in ("strandNames", "nodeNames"):
                    continue

            pixel_count = 0
            parm1 = m.get("parm1", "0")
            parm2 = m.get("parm2", "0")
            pixel_count_attr = m.get("PixelCount", "")
            if pixel_count_attr:
                try:
                    pixel_count = int(pixel_count_attr)
                except ValueError:
                    pass
            elif parm1 and parm2:
                try:
                    pixel_count = int(parm1) * int(parm2)
                except ValueError:
                    pass

            model = LightModel(
                name=m.get("name", ""),
                display_as=m.get("DisplayAs", ""),
                controller=m.get("Controller", ""),
                pixel_count=pixel_count,
                string_type=m.get("StringType", "RGB Nodes"),
                submodels=submodels,
                face_definitions=face_definitions,
            )
            models.append(model)

    logger.info(f"Loaded {len(models)} models from {effects_file}")
    return models


def load_model_groups(show_path: Path) -> list[ModelGroup]:
    """Parse model groups from xlights_rgbeffects.xml.

    Model groups live in their own top-level <modelGroups> element (a
    sibling of <models>, not nested inside it) -- and a group's "models"
    list can itself reference other *group* names, not just leaf model
    names (e.g. "Arches GRP" -> "Arches" -> "Arch 1,Arch 2,Arch 3,Arch 4").
    This resolves that nesting down to real model names so callers get a
    flat, usable member list.
    """
    effects_file = show_path / "xlights_rgbeffects.xml"
    if not effects_file.exists():
        return []

    tree = ET.parse(effects_file)
    root = tree.getroot()
    models_elem = root.find("models")
    real_model_names = {m.get("name", "") for m in (models_elem or []) if m.tag == "model"}
    # Parent model -> its real submodel names, so "Parent/SubName" references
    # in a group's member list can be checked against what actually exists
    # rather than assumed valid just because "Parent" exists. Genuinely stale
    # submodel refs do occur (e.g. a submodel renamed/removed since a group
    # was built) and silently accepting them produces an orphan Element that
    # renders as nothing when a sequence targets it.
    submodels_by_parent: dict[str, set[str]] = {}
    for m in (models_elem or []):
        if m.tag != "model":
            continue
        subs = {c.get("name", "") for c in m if c.tag == "subModel" and c.get("name")}
        if subs:
            submodels_by_parent[m.get("name", "")] = subs
    model_groups_elem = root.find("modelGroups")
    if model_groups_elem is None:
        return []

    raw: dict[str, tuple[str, list[str]]] = {}  # name -> (grid_size/layout meta unused here, raw member refs)
    for g in model_groups_elem:
        if g.tag != "modelGroup":
            continue
        name = g.get("name", "")
        members_str = g.get("models", "")
        refs = [n.strip() for n in members_str.split(",") if n.strip()]
        raw[name] = (g, refs)

    def resolve(name: str, seen: set[str]) -> list[str]:
        if name in seen or name not in raw:
            return []
        seen = seen | {name}
        _, refs = raw[name]
        out: list[str] = []
        for ref in refs:
            if ref in real_model_names:
                out.append(ref)
            elif ref in raw:
                out.extend(resolve(ref, seen))
            elif "/" in ref:
                # A submodel reference, e.g. "Front Center Window Matrix/Verticals"
                # -- xLights addresses submodels as "ParentModel/SubmodelName" and
                # groups reference them this way constantly (window outline
                # verticals/horizontals, snowflake arms/centers/rings/spokes).
                # Keep the full qualified name only if that submodel actually
                # exists on that parent right now.
                parent, sub = ref.split("/", 1)
                if sub in submodels_by_parent.get(parent, ()):
                    out.append(ref)
            # else: stale reference to a model/group/submodel that no longer
            # exists -- skip
        return out

    groups = []
    for name, (elem, _refs) in raw.items():
        members = resolve(name, set())
        group = ModelGroup(
            name=name,
            members=members,
            grid_size=elem.get("GridSize", ""),
            layout=elem.get("layout", ""),
        )
        groups.append(group)

    logger.info(f"Loaded {len(groups)} model groups from {effects_file}")
    return groups
