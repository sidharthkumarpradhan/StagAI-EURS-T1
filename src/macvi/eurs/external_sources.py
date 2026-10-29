"""Public underwater datasets with pixel masks, and how each maps onto the 5 EURS classes.

Researched 4 Oct 2026; sizes, licences and the reasoning behind every mapping are in
docs/research.md ("External training data"). Every source used must be disclosed at submission.

The EURS taxonomy (organisers' starter kit, ``taxonomy.py``): coral = names containing "coral";
elasmobranch = sharks and rays; other = turtle, starfish, seahorse; fish = every other annotated
animal (the catch-all); background = anything not annotated. The organisers' label map is not
public, so a foreign class whose EURS fate is unknowable (an octopus would be "fish" if EURS
annotated it, background if not) is mapped to ``IGNORE`` instead of guessed.

Masks written by the data kernel hold:

- 0..4: EURS class ids;
- ``UNLABELLED`` (254): pixels the source never annotated (instance and salient-object datasets
  label a few objects per image); training maps them to background or ignore per source;
- ``IGNORE`` (255): annotated, but outside the EURS taxonomy or ambiguous; never trained on.

Readers yield ``(item id, BGR image, mapped mask)``; nothing here touches the network.
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

BG, CORAL, FISH, ELASMO, OTHER = range(5)
UNLABELLED, IGNORE = 254, 255
# Paint order for overlapping instances (later wins): the starter kit's coral < fish < other <
# elasmobranch, with background-mapped and ignored objects below every EURS object.
PAINT_ORDER = (BG, IGNORE, CORAL, FISH, OTHER, ELASMO)

Item = tuple[str, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    kind: str                       # "coco_zip" | "index_zip" | "parquet"
    files: tuple[str, ...]          # direct download URLs (no login)
    licence: str
    licence_url: str
    citation: str
    mapping: dict = field(default_factory=dict)   # source class (name or id) -> EURS id / IGNORE
    annotations: str = ""           # coco_zip: substring selecting the annotation JSON files
    unlabelled: int = UNLABELLED    # value of pixels no annotation covers
    notes: str = ""


def _names(value: int, *names: str) -> dict[str, int]:
    return {n.lower(): value for n in names}


# USIS16K (158 categories, read from its COCO files on 4 Oct 2026); unknown names map to IGNORE.
USIS16K_MAPPING = {
    "background": BG,
    **_names(ELASMO, "MantaRay", "ElectricRay", "Sawfish", "BullheadShark", "GreatWhiteShark", "WhaleShark",
             "HammerheadShark", "ThresherShark"),
    **_names(OTHER, "LinckiaLaevigata", "OreasterReticulatus", "ProtoreasterNodosus", "Hippocampus", "Turtle"),
    **_names(CORAL, "FanCoral", "ElkhornCoral", "BrainCoral"),
    **_names(FISH, "MorayEel", "ObicularBatfish", "Lionfish", "Trumpetfish", "Flounder", "Frogfish", "Sailfish",
             "EnoplosusArmatus", "PseudanthiasPleurotaenia", "Mola", "MoorishIdol", "BicolorAngelfish",
             "AtlanticSpadefish", "SpottedDrum", "ThreespotAngelfish", "ChromisDimidiata", "RedseaBannerfish",
             "HeniochusVarius", "MaldivesDamselfish", "ScissortailSergeant", "FireGoby", "Twin-spotGoby",
             "Porcupinefish", "YellowBoxfish", "BlackspottedPuffer", "BlueParrotfish", "StoplightParrotfish",
             "PomacentrusSulfureus", "LunarFusilier", "OcellarisClownfish", "CinnamonClownfish", "RedSeaClownfish",
             "PinkAnemonefish", "OrangeSkunkClownfish", "GiantWrasse", "SpottedWrasse", "AnampsesTwistii",
             "Blue-spottedWrasse", "SlingjawWrasse", "Red-breastedWrasse", "PeacockGrouper", "PotatoGrouper",
             "Graysby", "RedmouthGrouper", "HumpbackGrouper", "CoralHind", "Porkfish", "AnyperodonLeucogrammicus",
             "WhitespottedSurgeonfish", "Orange-bandSurgeonfish", "ConvictSurgeonfish", "SohalSurgeonfish",
             "RegalBlueTang", "LinedSurgeonfish", "AchillesTang", "PowderBlueTang", "WhitecheekSurgeonfish",
             "SaddleButterflyfish", "MirrorButterflyfish", "BluecheekButterflyfish", "BlacktailButterflyfish",
             "RaccoonButterflyfish", "ThreadfinButterflyfish", "EritreanButterflyfish", "PyramidButterflyfish",
             "CopperbandButterflyfish"),
    # people, litter, wrecks, vehicles and ship parts: EURS never annotates them (background)
    **_names(BG, "Diver", "Swimmer", "PlasticBag", "PlasticBottle", "PlasticCup", "PlasticBox", "GlassBottle",
             "SurgicalMask", "Tyre", "Can", "Shipwreck", "WreckedAircraft", "WreckedCar", "WreckedTank", "Gun",
             "Phone", "Ring", "Boots", "Glasses", "Coin", "Statue", "Amphora", "Anchor", "ShipSWheel", "AUV", "ROV",
             "MilitarySubmarines", "PersonalSubmarines", "ShipSAnode", "OverBoardValve", "Propeller",
             "SeaChestGrating", "SubmarinePipeline", "PipelineSAnode"),
    # invertebrates, mammals, snakes, penguins, sea dragons: EURS fate unknown -> IGNORE (the default)
}

# Coralscapes id2label (HF EPFL-ECEO/coralscapes, id2label.json); 0 = not annotated.
CORALSCAPES_CORAL = (3, 4, 6, 16, 17, 19, 20, 21, 22, 23, 25, 27, 28, 31, 32, 33, 34, 36, 37)
CORALSCAPES_IGNORE = (11, 24, 26, 29, 30, 35, 39)  # other animal (starfish+turtles+worms), clams, cucumbers, ...
CORALSCAPES_MAPPING = {0: UNLABELLED, 9: FISH, 38: OTHER,  # 38 crown-of-thorns is a starfish
                       **{i: CORAL for i in CORALSCAPES_CORAL}, **{i: IGNORE for i in CORALSCAPES_IGNORE},
                       **{i: BG for i in (1, 2, 5, 7, 8, 10, 12, 13, 14, 15, 18)}}

HF = "https://huggingface.co/datasets"
SOURCES: dict[str, Source] = {s.name: s for s in (
    Source("usis16k", "USIS16K (Hong et al., 2025)", "coco_zip",
           ("https://drive.usercontent.google.com/download?id=14tbW3Ie8MfVjQy9DJKXnFlJ_g-6Z6xcX&export=download&confirm=t",),
           "free for academic research, not for commercial use (dataset README)",
           "https://github.com/LinHong-HIT/USIS16K",
           "L. Hong et al., USIS16K: High-Quality Dataset for Underwater Salient Instance Segmentation, arXiv:2506.19472",
           USIS16K_MAPPING, annotations="annotations/instances_",
           notes="salient instances only; the only source with sharks, rays, seahorses and starfish by species"),
    Source("uiis10k", "UIIS10K (Li et al., 2025)", "coco_zip", (f"{HF}/LiamLian0727/UIIS10K/resolve/main/UIIS10K.zip",),
           "Apache-2.0", f"{HF}/LiamLian0727/UIIS10K",
           "H. Li et al., Advancing Marine Research: UWSAM Framework and UIIS10K Dataset, arXiv:2505.15581",
           {"fish": FISH, "reptiles": OTHER, "corals": CORAL, "plants": BG, "ruins": BG, "garbage": BG,
            "human": BG, "robots": BG, "arthropoda": IGNORE, "mollusk": IGNORE},
           annotations="annotations/multiclass_", notes="instances; 'fish' includes sharks and rays"),
    Source("usis10k", "USIS10K (Lian et al., ICML 2024)", "coco_zip", (f"{HF}/LiamLian0727/USIS10K/resolve/main/USIS10K.zip",),
           "Apache-2.0", f"{HF}/LiamLian0727/USIS10K",
           "S. Lian et al., Diving into Underwater: SAM Guided Underwater Salient Instance Segmentation, ICML 2024",
           {"fish": FISH, "reefs": CORAL, "wrecks/ruins": BG, "aquatic plants": BG, "human divers": BG,
            "robots": BG, "sea-floor": BG},
           annotations="multi_class_annotations/", notes="salient instances; 'fish' includes sharks and rays"),
    Source("suim", "SUIM (Islam et al., IROS 2020)", "index_zip", (f"{HF}/SatwikKambham/suim/resolve/main/SUIM.zip",),
           "MIT", "https://github.com/xahidbuffon/SUIM",
           "M. J. Islam et al., Semantic Segmentation of Underwater Imagery: Dataset and Benchmark, IROS 2020",
           {0: BG, 1: BG, 2: BG, 3: BG, 4: BG, 5: CORAL, 6: FISH, 7: BG},
           notes="dense 8 classes (grey-index mirror of the original); 'fish and vertebrates' includes sharks, "
                 "rays and turtles, 'reefs and invertebrates' maps to coral"),
    Source("coralscapes", "Coralscapes (Sauder et al., 2025)", "parquet",
           tuple(f"{HF}/EPFL-ECEO/coralscapes/resolve/main/data/{split}-{i:05d}-of-{n:05d}.parquet"
                 for split, n in (("train", 9), ("validation", 1), ("test", 3)) for i in range(n)),
           "Apache-2.0", f"{HF}/EPFL-ECEO/coralscapes",
           "J. Sauder et al., The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs, arXiv:2503.20000",
           CORALSCAPES_MAPPING, notes="dense 39 benthic classes, Red Sea reef transects (mostly coral and substrate)"),
)}
DEFAULT_ORDER = ("usis16k", "uiis10k", "usis10k", "suim", "coralscapes")  # dedupe keeps the first copy


# ------------------------------------------------------------------ mapping helpers (pure)

def lookup(mapping: dict, key) -> int:
    """EURS id of a source class: names case-insensitively (trailing spaces ignored), ids exactly;
    anything unmapped is IGNORE."""
    if isinstance(key, str):
        return mapping.get(key.strip().lower(), IGNORE)
    return mapping.get(int(key), IGNORE)


def remap(index_mask: np.ndarray, mapping: dict) -> np.ndarray:
    """Map an index mask (source ids) through ``mapping``; ids not listed become IGNORE."""
    lut = np.full(256, IGNORE, np.uint8)
    for k, v in mapping.items():
        lut[int(k)] = v
    return lut[index_mask]


def paint(shape: tuple[int, int], objects: list[tuple[int, list[list[float]]]], unlabelled: int = UNLABELLED) -> np.ndarray:
    """Rasterise ``(EURS id, COCO polygons)`` objects in ``PAINT_ORDER`` onto an unlabelled canvas."""
    mask = np.full(shape, unlabelled, np.uint8)
    rank = {v: i for i, v in enumerate(PAINT_ORDER)}
    for value, polygons in sorted(objects, key=lambda o: rank.get(o[0], 0)):
        pts = [np.round(np.asarray(p, np.float64).reshape(-1, 2)).astype(np.int32) for p in polygons if len(p) >= 6]
        if pts:
            cv2.fillPoly(mask, pts, int(value))
    return mask


def decode(data: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if img is None:
        raise ValueError("cannot decode image")
    return img


# ------------------------------------------------------------------ readers

def read_coco_zip(path: Path, src: Source, limit: int | None = None) -> Iterator[Item]:
    """Images and COCO polygon instances of every annotation file whose name contains
    ``src.annotations``; an image is found by its base name (the split folder breaks ties)."""
    with zipfile.ZipFile(path) as zf:
        members = defaultdict(list)
        for n in zf.namelist():
            if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                members[n.rsplit("/", 1)[-1]].append(n)
        n_out = 0
        for ann_name in sorted(n for n in zf.namelist() if src.annotations in n and n.endswith(".json")):
            coco = json.loads(zf.read(ann_name))
            split = next((s for s in ("train", "val", "test") if s in ann_name.rsplit("/", 1)[-1]), "")
            cats = {str(c["id"]): lookup(src.mapping, c["name"]) for c in coco["categories"]}
            objects = defaultdict(list)
            for a in coco["annotations"]:
                if isinstance(a.get("segmentation"), list) and a["segmentation"]:  # RLE (crowd) skipped
                    objects[a["image_id"]].append((cats.get(str(a["category_id"]), IGNORE), a["segmentation"]))
            for im in coco["images"]:
                base = im["file_name"].rsplit("/", 1)[-1]
                options = members.get(base, [])
                if not options:
                    continue
                member = next((m for m in options if f"/{split}/" in m), options[0])
                img = decode(zf.read(member))
                if img.shape[:2] != (im["height"], im["width"]):
                    raise ValueError(f"{src.name}: {member} is {img.shape[:2]}, annotation says {(im['height'], im['width'])}")
                stem = Path(base).stem
                item_id = stem if not split or stem.startswith(split) else f"{split}_{stem}"
                yield item_id, img, paint(img.shape[:2], objects[im["id"]], src.unlabelled)
                n_out += 1
                if limit and n_out >= limit:
                    return


def read_index_zip(path: Path, src: Source, limit: int | None = None) -> Iterator[Item]:
    """Images in ``*/images/`` paired by stem with grey index masks in ``*/masks/`` (SUIM)."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        masks = {n.rsplit("/", 1)[-1].rsplit(".", 1)[0]: n for n in names if "/masks/" in n and not n.endswith("/")}
        n_out = 0
        for n in sorted(x for x in names if "/images/" in x and not x.endswith("/")):
            stem = n.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if stem not in masks:
                continue
            img, raw = decode(zf.read(n)), decode(zf.read(masks[stem]), cv2.IMREAD_UNCHANGED)
            if raw.ndim == 3:
                raise ValueError(f"{src.name}: {masks[stem]} is a colour mask, expected grey class ids")
            if raw.shape[:2] != img.shape[:2]:
                raw = cv2.resize(raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            split = "test" if "/test/" in n.lower() else "train"
            yield f"{split}_{stem}", img, remap(raw, src.mapping)
            n_out += 1
            if limit and n_out >= limit:
                return


def read_parquet(path: Path, src: Source, limit: int | None = None) -> Iterator[Item]:
    """Rows of a Hugging Face image-segmentation parquet: ``image`` and ``label`` structs with PNG/JPEG bytes."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_out = 0
    for g in range(pf.num_row_groups):
        table = pf.read_row_group(g, columns=["image", "label"]).to_pylist()
        for i, row in enumerate(table):
            img = decode(row["image"]["bytes"])
            raw = decode(row["label"]["bytes"], cv2.IMREAD_UNCHANGED)
            if raw.ndim == 3:
                raw = raw[..., 0]
            name = Path(row["image"].get("path") or f"{path.stem}_{g}_{i}").stem
            yield f"{path.stem.split('-')[0]}_{name}", img, remap(raw, src.mapping)
            n_out += 1
            if limit and n_out >= limit:
                return


READERS: dict[str, Callable[[Path, Source, int | None], Iterator[Item]]] = {
    "coco_zip": read_coco_zip, "index_zip": read_index_zip, "parquet": read_parquet}


def items(src: Source, files: list[Path], limit: int | None = None) -> Iterator[Item]:
    """Every item of ``src`` from its downloaded ``files`` (``limit``: at most this many per file)."""
    for f in files:
        yield from READERS[src.kind](f, src, limit)


def mapping_table() -> list[dict]:
    """Rows for docs and the manifest: source class -> EURS class name."""
    names = {BG: "background", CORAL: "coral", FISH: "fish", ELASMO: "elasmobranch", OTHER: "other",
             IGNORE: "ignore", UNLABELLED: "unlabelled"}
    return [{"source": s.name, "class": str(k), "eurs": names[v]} for s in SOURCES.values() for k, v in s.mapping.items()]
