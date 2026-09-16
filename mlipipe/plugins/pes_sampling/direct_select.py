"""Bundled local runner for MAML/DIRECT representative-structure selection.

This module adapts the reviewed ``direct_sampling/direct.py`` implementation into
an explicit, non-destructive MLIPipe command.  Structure parsing and output
normalization live here; the selection algorithm remains MAML's DIRECTSampler.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence


@dataclass
class StructRecord:
    structure: Any
    source_file: Path
    frame_index: int
    source_kind: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def parse_lammps_type_map(text: Optional[str]) -> Optional[dict[int, str]]:
    if text is None:
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("--lammps-type-map must be a JSON object") from exc
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("--lammps-type-map must be a non-empty JSON object")

    parsed: dict[int, str] = {}
    for key, value in raw.items():
        try:
            atom_type = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("LAMMPS atom types must be positive integers") from exc
        if atom_type < 1 or not isinstance(value, str) or not re.fullmatch(r"[A-Z][a-z]?", value):
            raise ValueError(
                "LAMMPS atom types must be positive integers mapped to element symbols"
            )
        if atom_type in parsed:
            raise ValueError("LAMMPS atom types must be unique after integer conversion")
        parsed[atom_type] = value
    return parsed


def atoms_to_structure_with_optional_lammps_map(
    atoms: Any, lammps_type_map: Optional[dict[int, str]]
) -> Any:
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except Exception as exc:
        raise RuntimeError("pymatgen with ASE support is required") from exc

    if lammps_type_map is not None:
        type_array = None
        for key in ("type", "types", "atom_types"):
            if key in atoms.arrays:
                type_array = atoms.arrays[key]
                break
        if type_array is None:
            raise RuntimeError(
                "a LAMMPS type map was provided, but the ASE atoms object has no "
                "type/types/atom_types array"
            )

        symbols = []
        for value in type_array:
            atom_type = int(value)
            if atom_type not in lammps_type_map:
                raise RuntimeError("LAMMPS atom type %d is absent from lammps_type_map" % atom_type)
            symbols.append(lammps_type_map[atom_type])
        atoms = atoms.copy()
        atoms.set_chemical_symbols(symbols)

    return AseAtomsAdaptor.get_structure(atoms)


def read_vasprun_structures(
    path: Path, stride: int, max_frames: Optional[int]
) -> List[StructRecord]:
    try:
        from pymatgen.io.vasp import Vasprun
    except Exception as exc:
        raise RuntimeError("pymatgen is required for vasprun inputs") from exc

    vasprun = Vasprun(
        str(path),
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
        exception_on_bad_xml=False,
    )
    records: List[StructRecord] = []
    kept = 0
    for ionic_index, step in enumerate(vasprun.ionic_steps):
        if ionic_index % stride != 0:
            continue
        structure = step.get("structure")
        if structure is None:
            continue
        records.append(StructRecord(structure, path, ionic_index, "vasprun"))
        kept += 1
        if max_frames is not None and kept >= max_frames:
            break

    final_structure = getattr(vasprun, "final_structure", None)
    if not records and final_structure is not None:
        records.append(StructRecord(final_structure, path, -1, "vasprun_final"))
    return records


def read_xdatcar_structures(
    path: Path, stride: int, max_frames: Optional[int]
) -> List[StructRecord]:
    try:
        from pymatgen.io.vasp import Xdatcar
    except Exception as exc:
        raise RuntimeError("pymatgen is required for XDATCAR inputs") from exc

    records: List[StructRecord] = []
    kept = 0
    for frame_index, structure in enumerate(Xdatcar(str(path)).structures):
        if frame_index % stride != 0:
            continue
        records.append(StructRecord(structure, path, frame_index, "xdatcar"))
        kept += 1
        if max_frames is not None and kept >= max_frames:
            break
    return records


def read_ase_structures(
    path: Path,
    stride: int,
    max_frames: Optional[int],
    lammps_type_map: Optional[dict[int, str]],
) -> List[StructRecord]:
    try:
        from ase.io import iread
    except Exception as exc:
        raise RuntimeError("ASE is required for .traj/.xyz/.extxyz/LAMMPS dump inputs") from exc

    suffix = path.suffix.lower()
    is_lammps = suffix in (".dump", ".lammpstrj") or path.name.startswith("dump.")
    kwargs = {"format": "lammps-dump-text"} if is_lammps else {}
    records: List[StructRecord] = []
    kept = 0
    for frame_index, atoms in enumerate(iread(str(path), index=":", **kwargs)):
        if frame_index % stride != 0:
            continue
        structure = atoms_to_structure_with_optional_lammps_map(atoms, lammps_type_map)
        records.append(
            StructRecord(
                structure,
                path,
                frame_index,
                "lammps_dump" if is_lammps else "ase",
            )
        )
        kept += 1
        if max_frames is not None and kept >= max_frames:
            break
    return records


def read_single_structure(path: Path) -> List[StructRecord]:
    try:
        from pymatgen.core import Structure
    except Exception as exc:
        raise RuntimeError("pymatgen is required for structure inputs") from exc
    return [StructRecord(Structure.from_file(str(path)), path, 0, "single_structure")]


def detect_and_read_structures(
    path: Path,
    stride: int,
    max_frames: Optional[int],
    lammps_type_map: Optional[dict[int, str]],
) -> List[StructRecord]:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in ("vasprun.xml", "vasprun.xml.gz") or name.endswith("vasprun.xml.gz"):
        return read_vasprun_structures(path, stride, max_frames)
    if name.startswith("xdatcar"):
        return read_xdatcar_structures(path, stride, max_frames)
    if suffix in (".traj", ".xyz", ".extxyz", ".dump", ".lammpstrj") or path.name.startswith(
        "dump."
    ):
        return read_ase_structures(path, stride, max_frames, lammps_type_map)
    return read_single_structure(path)


def gather_files(input_dirs: Sequence[Path], globs: Sequence[str], recursive: bool) -> List[Path]:
    seen: set[Path] = set()
    files: List[Path] = []
    for root in input_dirs:
        if not root.is_dir():
            raise FileNotFoundError("input directory not found: %s" % root.resolve())
        for pattern in globs:
            iterator = root.rglob(pattern) if recursive else root.glob(pattern)
            for candidate in sorted(iterator):
                if candidate.is_file():
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        files.append(candidate)
    if not files:
        roots = ", ".join(str(path.resolve()) for path in input_dirs)
        raise RuntimeError("no files matched %s under %s" % (list(globs), roots))
    return files


def load_records(
    files: Sequence[Path],
    stride: int,
    max_frames_per_file: Optional[int],
    max_input_structures: Optional[int],
    lammps_type_map: Optional[dict[int, str]],
) -> List[StructRecord]:
    records: List[StructRecord] = []
    skipped_files = 0
    for file_index, path in enumerate(files, start=1):
        try:
            new_records = detect_and_read_structures(
                path, stride, max_frames_per_file, lammps_type_map
            )
            for record in new_records:
                if not record.structure.is_ordered:
                    raise RuntimeError(
                        "disordered structure found: %s, frame=%d"
                        % (record.source_file, record.frame_index)
                    )
            records.extend(new_records)
            print(
                "[INFO] %d/%d loaded %d structures from %s"
                % (file_index, len(files), len(new_records), path)
            )
        except Exception as exc:
            skipped_files += 1
            print("[WARN] skip file %s: %s" % (path, exc))

        if max_input_structures is not None and len(records) >= max_input_structures:
            records = records[:max_input_structures]
            print("[INFO] Reached max_input_structures = %d" % max_input_structures)
            break

    if not records:
        raise RuntimeError("no structures loaded; check the input files and formats")
    print("[INFO] Loaded structures: %d" % len(records))
    print("[INFO] Skipped files: %d" % skipped_files)
    return records


def run_direct(
    records: Sequence[StructRecord],
    n_clusters: int,
    threshold_init: float,
    k_per_cluster: int,
) -> tuple[Any, dict[str, Any], List[int], int]:
    try:
        from maml.sampling.direct import (
            BirchClustering,
            DIRECTSampler,
            SelectKFromClusters,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to import MAML DIRECT modules; install a compatible maml runtime"
        ) from exc

    effective_clusters = min(n_clusters, len(records))
    if effective_clusters != n_clusters:
        print(
            "[WARN] n_clusters=%d exceeds structure count=%d; using %d"
            % (n_clusters, len(records), effective_clusters)
        )
    sampler = DIRECTSampler(
        clustering=BirchClustering(
            n=effective_clusters,
            threshold_init=threshold_init,
        ),
        select_k_from_clusters=SelectKFromClusters(k=k_per_cluster),
    )
    selection = sampler.fit_transform([record.structure for record in records])
    if not isinstance(selection, Mapping) or "selected_indexes" not in selection:
        keys = list(selection) if isinstance(selection, Mapping) else []
        raise RuntimeError("unexpected DIRECT output keys: %s" % keys)
    indexes = sorted(int(value) for value in selection["selected_indexes"])
    if not indexes:
        raise RuntimeError("DIRECT selected no structures")
    if len(indexes) != len(set(indexes)):
        raise RuntimeError("DIRECT returned duplicate selected indexes")
    if indexes[0] < 0 or indexes[-1] >= len(records):
        raise RuntimeError("DIRECT returned an out-of-range selected index")
    return sampler, dict(selection), indexes, effective_clusters


def safe_formula(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]", "_", value)
    return cleaned or "structure"


def write_selected_outputs(
    records: Sequence[StructRecord], selected_indexes: Sequence[int], out_dir: Path
) -> Path:
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("output directory already exists: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = out_dir / "manifest.csv"

    try:
        from pymatgen.io.vasp import Poscar
    except Exception as exc:
        raise RuntimeError("pymatgen is required to write selected POSCAR files") from exc

    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "selected_order",
                "input_index",
                "source_kind",
                "source_file",
                "frame_index",
                "output_file",
                "formula",
            ]
        )
        for selected_order, index in enumerate(selected_indexes, start=1):
            record = records[index]
            formula = str(record.structure.composition.reduced_formula)
            output_name = "%05d_%s.vasp" % (selected_order, safe_formula(formula))
            Poscar(record.structure).write_file(str(out_dir / output_name))
            writer.writerow(
                [
                    selected_order,
                    index,
                    record.source_kind,
                    str(record.source_file.resolve()),
                    record.frame_index,
                    output_name,
                    formula,
                ]
            )
    print("[INFO] Wrote selected POSCAR files to: %s" % out_dir.resolve())
    print("[INFO] Wrote manifest to: %s" % manifest_path.resolve())
    return manifest_path


def write_direct_pca_plots(
    direct_sampler: Any,
    direct_selection: Mapping[str, Any],
    selected_indexes: Sequence[int],
    out_dir: Path,
) -> None:
    """Write the same optional Plotly diagnostics as the reviewed implementation."""

    try:
        import numpy as np
        import plotly.graph_objects as go
    except Exception as exc:
        print("[WARN] Skip PCA plots because optional dependencies are missing: %s" % exc)
        return
    if "PCAfeatures" not in direct_selection:
        print("[WARN] DIRECT output has no PCAfeatures; skip PCA plots")
        return

    pca_features = np.asarray(direct_selection["PCAfeatures"])
    selected = np.asarray(selected_indexes, dtype=int)
    if pca_features.ndim != 2 or pca_features.shape[1] < 2:
        print("[WARN] PCAfeatures shape is %s; skip PCA plots" % (pca_features.shape,))
        return

    explained_variance = np.asarray(direct_sampler.pca.pca.explained_variance_ratio_)
    n_components = pca_features.shape[1]
    usable_variance = explained_variance[:n_components]
    usable_variance = np.where(np.isclose(usable_variance, 0.0), 1.0, usable_variance)
    unweighted = pca_features / usable_variance
    selected_features = unweighted[selected]
    selected_mask = np.zeros(len(unweighted), dtype=bool)
    selected_mask[selected] = True

    n_show = min(30, len(explained_variance))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(range(1, n_show + 1)),
            y=explained_variance[:n_show] * 100,
            mode="lines+markers",
            name="Explained variance",
        )
    )
    figure.update_layout(
        title="DIRECT PCA explained variance",
        xaxis_title="PC index",
        yaxis_title="Explained variance (%)",
        template="plotly_white",
    )
    figure.write_html(str(out_dir / "direct_pca_explained_variance.html"))
    try:
        figure.write_image(str(out_dir / "direct_pca_explained_variance.png"), scale=3)
    except Exception:
        pass

    all_indexes = np.arange(len(unweighted))
    unselected = all_indexes[~selected_mask]
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=unweighted[unselected, 0],
            y=unweighted[unselected, 1],
            mode="markers",
            name="Not selected %d" % len(unselected),
            marker=dict(size=4, opacity=0.35),
            customdata=unselected,
            hovertemplate=("input_index=%{customdata}<br>PC1=%{x}<br>PC2=%{y}<extra></extra>"),
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=selected_features[:, 0],
            y=selected_features[:, 1],
            mode="markers",
            name="DIRECT selected %d" % len(selected_features),
            marker=dict(size=7, opacity=0.85, symbol="x"),
            customdata=selected,
            hovertemplate=("input_index=%{customdata}<br>PC1=%{x}<br>PC2=%{y}<extra></extra>"),
        )
    )
    figure.update_layout(
        title="DIRECT sampling coverage in PCA feature space",
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        template="plotly_white",
    )
    figure.write_html(str(out_dir / "direct_pca_coverage.html"))
    try:
        figure.write_image(str(out_dir / "direct_pca_coverage.png"), scale=3)
    except Exception:
        pass

    bins_count = min(1000, max(50, len(unweighted) // 2))
    scores = []
    for index in range(n_components):
        all_values = unweighted[:, index]
        selected_values = all_values[selected]
        minimum = np.min(all_values)
        maximum = np.max(all_values)
        if np.isclose(minimum, maximum):
            scores.append(1.0)
            continue
        bins = np.linspace(minimum, maximum, bins_count)
        populated_all = np.count_nonzero(np.histogram(all_values, bins=bins)[0])
        populated_selected = np.count_nonzero(np.histogram(selected_values, bins=bins)[0])
        scores.append(populated_selected / populated_all if populated_all else 0.0)

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=["PC %d" % (index + 1) for index in range(len(scores))],
            y=scores,
            name="DIRECT, mean coverage score = %.3f" % np.mean(scores),
        )
    )
    figure.update_layout(
        title="DIRECT feature coverage score",
        xaxis_title="Principal component",
        yaxis_title="Coverage score",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
    )
    figure.write_html(str(out_dir / "direct_feature_coverage_score.html"))
    try:
        figure.write_image(str(out_dir / "direct_feature_coverage_score.png"), scale=3)
    except Exception:
        pass
    print("[INFO] Wrote DIRECT Plotly PCA plots to: %s" % out_dir.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run bundled MAML DIRECT selection from POSCAR/vasprun/XDATCAR/ASE/LAMMPS inputs."
        )
    )
    parser.add_argument("--input-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--globs", nargs="+", required=True)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--stride", required=True, type=positive_int)
    parser.add_argument("--max-frames-per-file", type=positive_int)
    parser.add_argument("--max-input-structures", type=positive_int)
    parser.add_argument("--n-clusters", required=True, type=positive_int)
    parser.add_argument("--threshold-init", required=True, type=positive_float)
    parser.add_argument("--k-per-cluster", required=True, type=positive_int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lammps-type-map")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out_dir.exists() or args.out_dir.is_symlink():
        raise FileExistsError("output directory already exists: %s" % args.out_dir)
    lammps_type_map = parse_lammps_type_map(args.lammps_type_map)
    files = gather_files(args.input_dirs, args.globs, not args.no_recursive)
    print("[INFO] Matched files: %d" % len(files))
    print("[INFO] stride = %d" % args.stride)
    print("[INFO] max_frames_per_file = %s" % args.max_frames_per_file)
    print("[INFO] max_input_structures = %s" % args.max_input_structures)
    records = load_records(
        files,
        args.stride,
        args.max_frames_per_file,
        args.max_input_structures,
        lammps_type_map,
    )
    sampler, selection, selected_indexes, effective_clusters = run_direct(
        records,
        args.n_clusters,
        args.threshold_init,
        args.k_per_cluster,
    )
    print("[RESULT] DIRECT selected: %d / %d" % (len(selected_indexes), len(records)))
    print(
        "[RESULT] Upper bound about n_clusters*k = %d" % (effective_clusters * args.k_per_cluster)
    )
    write_selected_outputs(records, selected_indexes, args.out_dir)
    try:
        write_direct_pca_plots(sampler, selection, selected_indexes, args.out_dir)
    except Exception as exc:
        print("[WARN] Skip optional PCA plots after plotting error: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
