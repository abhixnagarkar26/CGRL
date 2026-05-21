"""
cap_assign.py
─────────────────────────────────────────────────────────────
Standalone cap-assignment utility.

Use this when you already have a boolean-subtracted mesh and
just need to re-stamp ModelFaceIDs + write a .facenames file.

Inputs
  BASE_VTP        — original tree mesh (used only to read cap
                    geometry + ModelFaceID cell data)
  OLD_FACENAMES   — .facenames file that belongs to BASE_VTP
  RESULT_VTP      — mesh that needs face IDs assigned
                    (no ModelFaceID required on this mesh)

Outputs
  OUTPUT_VTP      — RESULT_VTP with ModelFaceID cell data added
  OUTPUT_FACENAMES — matching .facenames file
"""

import re
import sys
import numpy as np
import pyvista as pv


# ============================================================
# INPUTS  ← edit these
# ============================================================

BASE_VTP = r"C:\MS BME- Applied Study\CGRL\MODELS\trial\Tree.vtp"

OLD_FACENAMES = r"C:\MS BME- Applied Study\CGRL\MODELS\trial\Tree.vtp.facenames"

RESULT_VTP = r"C:\MS BME- Applied Study\CGRL\MODELS\trial\Result.vtp"

OUTPUT_VTP = r"C:\MS BME- Applied Study\CGRL\MODELS\trial\Result_assigned.vtp"

OUTPUT_FACENAMES = r"C:\MS BME- Applied Study\CGRL\MODELS\trial\Result_assigned.vtp.facenames"


# ============================================================
# TUNING
# ============================================================

# Percentile used to estimate cap radius from base mesh points
CAP_PERCENTILE = 95

# Scale factor on radius -> plane thickness tolerance.
# Raise (e.g. 0.001) if caps are under-assigned.
# Lower (e.g. 0.00001) if wall cells bleed into caps.
PLANE_TOL_FACTOR = 0.00005


# ============================================================
# FACENAME I/O
# ============================================================

def read_facenames(fname):
    faces = {}
    pattern = re.compile(r"set\s+gPolyDataFaceNames\((\d+)\)\s+(\S+)")
    with open(fname) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                fid  = int(m.group(1))
                name = m.group(2).strip().strip("{}")
                faces[fid] = name
    return faces


def write_facenames(path, assignments):
    with open(path, "w") as f:
        f.write("global gPolyDataFaceNames\n\n")
        for fid in sorted(assignments):
            f.write(f"set gPolyDataFaceNames({fid}) {assignments[fid]}\n")


# ============================================================
# SURFACE LOADING
# ============================================================

def load_surface(path):
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh


# ============================================================
# EXTRACT CAP GEOMETRY FROM BASE MESH
# ============================================================

def extract_cap_info(surface, facename_file, percentile=95):
    """
    Reads ModelFaceID cell data from `surface` and the matching
    .facenames file, then fits a plane + radius to every non-wall
    face region.  Returns a list of cap descriptor dicts.
    """
    face_map = read_facenames(facename_file)

    cap_entries = {
        fid: name
        for fid, name in face_map.items()
        if "wall" not in name.lower()
    }

    if not cap_entries:
        sys.exit("ERROR: No non-wall faces found in .facenames file.")

    if "ModelFaceID" not in surface.cell_data:
        sys.exit("ERROR: ModelFaceID cell data missing from base mesh.")

    mf_ids = surface.cell_data["ModelFaceID"]
    caps   = []

    for fid, name in cap_entries.items():
        cell_ids = np.where(mf_ids == fid)[0]

        if len(cell_ids) == 0:
            print(f"  WARNING: cap '{name}' (fid={fid}) has no cells — skipping")
            continue

        sub = surface.extract_cells(cell_ids)
        pts = np.array(sub.points)

        # Centroid
        centroid = pts.mean(axis=0)

        # PCA normal (last singular vector = smallest variance = plane normal)
        centered  = pts - centroid
        _, _, vh  = np.linalg.svd(centered)
        normal    = vh[-1]
        normal    = normal / np.linalg.norm(normal)

        # Radial extent within the cap plane
        vec             = pts - centroid
        plane_component = np.outer(np.dot(vec, normal), normal)
        projected       = vec - plane_component
        radial_dist     = np.linalg.norm(projected, axis=1)
        radius          = np.percentile(radial_dist, percentile)
        plane_tol       = radius * PLANE_TOL_FACTOR

        caps.append({
            "fid":       fid,
            "name":      name,
            "centroid":  centroid,
            "normal":    normal,
            "radius":    radius,
            "plane_tol": plane_tol,
        })

        print(
            f"  Cap '{name}' | fid={fid} | "
            f"radius={radius:.4f} | plane_tol={plane_tol:.6f}"
        )

    return caps


# ============================================================
# ASSIGN FACE IDs TO RESULT MESH
# ============================================================

def assign_face_ids(result_mesh, caps):
    """
    Stamps ModelFaceID onto every cell of `result_mesh`.
    Cells that fall within a cap's plane+radius envelope get that
    cap's fid; all remaining cells become wall (max_fid + 1).

    Returns (stamped_mesh, assignments_dict).
    """
    tri   = result_mesh.triangulate()
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    pts   = tri.points

    # Vectorised cell centroids
    cell_centroids = pts[faces].mean(axis=1)
    face_id_array  = np.zeros(tri.n_cells, dtype=int)

    for cap in caps:
        centroid  = cap["centroid"]
        normal    = cap["normal"]
        radius    = cap["radius"]
        plane_tol = cap["plane_tol"]

        vec         = cell_centroids - centroid
        plane_dist  = np.abs(np.dot(vec, normal))
        projected   = vec - np.outer(np.dot(vec, normal), normal)
        radial_dist = np.linalg.norm(projected, axis=1)

        mask = (plane_dist <= plane_tol) & (radial_dist <= radius)
        face_id_array[mask] = cap["fid"]

        n_assigned = mask.sum()
        status     = "✔" if n_assigned > 0 else "⚠ ZERO cells — check PLANE_TOL_FACTOR"
        print(f"  Cap '{cap['name']}' -> {n_assigned} cells  {status}")

    # Unassigned cells = wall
    wall_fid  = max(c["fid"] for c in caps) + 1
    wall_mask = face_id_array == 0
    face_id_array[wall_mask] = wall_fid
    print(f"  Wall (fid={wall_fid}) -> {wall_mask.sum()} cells")

    tri.cell_data["ModelFaceID"] = face_id_array

    assignments = {c["fid"]: c["name"] for c in caps}
    assignments[wall_fid] = "wall"

    return tri, assignments


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CAP ASSIGNMENT UTILITY")
    print("=" * 60)
    print(f"Base mesh   : {BASE_VTP}")
    print(f"Facenames   : {OLD_FACENAMES}")
    print(f"Result mesh : {RESULT_VTP}")
    print(f"Output VTP  : {OUTPUT_VTP}")
    print(f"Output fnames: {OUTPUT_FACENAMES}")
    print("=" * 60)

    # ----------------------------------------------------------
    # Load meshes
    # ----------------------------------------------------------
    print("\nLoading base mesh (for cap geometry)...")
    base = load_surface(BASE_VTP)
    print(f"  {base.n_points} points | {base.n_cells} cells")

    print("\nLoading result mesh (to be labelled)...")
    result = load_surface(RESULT_VTP)
    print(f"  {result.n_points} points | {result.n_cells} cells")

    # ----------------------------------------------------------
    # Extract cap planes from base
    # ----------------------------------------------------------
    print("\nExtracting cap geometry from base mesh...")
    caps = extract_cap_info(base, OLD_FACENAMES, percentile=CAP_PERCENTILE)
    print(f"Detected {len(caps)} cap(s).")

    if not caps:
        sys.exit("ERROR: No caps were extracted. Check .facenames file.")

    # ----------------------------------------------------------
    # Assign face IDs to result
    # ----------------------------------------------------------
    print("\nAssigning ModelFaceIDs to result mesh...")
    result_labelled, assignments = assign_face_ids(result, caps)

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    print(f"\nWriting labelled VTP  -> {OUTPUT_VTP}")
    result_labelled.save(OUTPUT_VTP)

    print(f"Writing .facenames    -> {OUTPUT_FACENAMES}")
    write_facenames(OUTPUT_FACENAMES, assignments)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\nFinal ModelFaceID assignments:")
    for fid in sorted(assignments):
        print(f"  ModelFaceID={fid}  ->  {assignments[fid]}")

    print("\nDONE ✔")