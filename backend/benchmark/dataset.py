import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class VerificationPair:
    first: Path
    second: Path
    same_person: bool


@dataclass(frozen=True)
class IdentitySplit:
    name: str
    enrollment: Path
    probe: Path


def _image_path(dataset_root: Path, name: str, image_number: str) -> Path:
    return dataset_root / name / f"{name}_{int(image_number):04d}.jpg"


def load_verification_pairs(
    pairs_file: Path,
    dataset_root: Path,
    maximum_per_class: int,
    seed: int,
) -> List[VerificationPair]:
    genuine: List[VerificationPair] = []
    impostor: List[VerificationPair] = []
    with pairs_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            values = [item.strip() for item in row]
            if len(values) < 3 or not values[0]:
                continue
            if len(values) >= 4 and values[3]:
                pair = VerificationPair(
                    first=_image_path(dataset_root, values[0], values[1]),
                    second=_image_path(dataset_root, values[2], values[3]),
                    same_person=False,
                )
                impostor.append(pair)
            else:
                pair = VerificationPair(
                    first=_image_path(dataset_root, values[0], values[1]),
                    second=_image_path(dataset_root, values[0], values[2]),
                    same_person=True,
                )
                genuine.append(pair)

    rng = random.Random(seed)
    rng.shuffle(genuine)
    rng.shuffle(impostor)
    selected = genuine[:maximum_per_class] + impostor[:maximum_per_class]
    rng.shuffle(selected)
    return selected


def collect_identity_splits(
    dataset_root: Path,
    maximum_identities: int,
    maximum_unknowns: int,
    seed: int,
) -> Tuple[List[IdentitySplit], List[Path]]:
    multi_image: List[IdentitySplit] = []
    single_image: List[Path] = []
    person_directories = sorted(
        path for path in dataset_root.iterdir() if path.is_dir()
    )
    rng = random.Random(seed)
    rng.shuffle(person_directories)
    for person_directory in person_directories:
        images = sorted(
            path
            for path in person_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if len(images) >= 2 and len(multi_image) < maximum_identities:
            multi_image.append(
                IdentitySplit(
                    name=person_directory.name,
                    enrollment=images[0],
                    probe=images[1],
                )
            )
        elif len(images) == 1 and len(single_image) < maximum_unknowns:
            single_image.append(images[0])
        if (
            len(multi_image) >= maximum_identities
            and len(single_image) >= maximum_unknowns
        ):
            break

    return multi_image, single_image


def unique_image_paths(
    pairs: Sequence[VerificationPair],
    identities: Sequence[IdentitySplit],
    unknowns: Sequence[Path],
) -> List[Path]:
    paths: Dict[str, Path] = {}
    for pair in pairs:
        paths[str(pair.first)] = pair.first
        paths[str(pair.second)] = pair.second
    for identity in identities:
        paths[str(identity.enrollment)] = identity.enrollment
        paths[str(identity.probe)] = identity.probe
    for path in unknowns:
        paths[str(path)] = path
    return [paths[key] for key in sorted(paths)]
