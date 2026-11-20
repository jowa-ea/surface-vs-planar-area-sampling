"""
Planar and triangulated surface area per pixel on a geographic (EPSG:4326) grid.

All computations use the exact per-row WGS84 ellipsoidal ground spacing of the
DEM grid, so no raster is reprojected before area is computed (paper, Section
"Ground Spacing on the Geographic Grid").
"""

import numpy as np

# WGS84 ellipsoid
A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)

# Elevations at or below this value are treated as voids. Copernicus DEM
# GLO-30 is distributed void-filled; this is a safeguard only.
ELEVATION_VOID_THRESHOLD = -1000.0


def row_scale_factors(lat_deg):
    """Ground distance (m) per degree of longitude and latitude at `lat_deg`.

    M is the meridional radius of curvature (North-South spacing), N the
    prime-vertical radius of curvature (East-West spacing, via N cos(phi)).
    """
    lat_rad = np.radians(lat_deg)
    sin2 = np.sin(lat_rad) ** 2
    M = A_WGS84 * (1 - E2_WGS84) / (1 - E2_WGS84 * sin2) ** 1.5
    N = A_WGS84 / np.sqrt(1 - E2_WGS84 * sin2)
    dy_per_deg = M * (np.pi / 180.0)
    dx_per_deg = N * np.cos(lat_rad) * (np.pi / 180.0)
    return dx_per_deg, dy_per_deg


def ground_spacing_m(lat_centers, pixel_size_deg_lon, pixel_size_deg_lat):
    """Per-row East-West and North-South pixel spacing (m), each shaped
    (rows, 1) so it broadcasts across columns."""
    dx_per_deg, dy_per_deg = row_scale_factors(lat_centers)
    dx = (dx_per_deg * pixel_size_deg_lon).reshape(-1, 1)
    dy = (dy_per_deg * pixel_size_deg_lat).reshape(-1, 1)
    return dx, dy


def pixel_area_m2(dx, dy, rows, cols):
    """Planar (ellipsoidal) pixel area a_p = dx * dy, shape (rows, cols)."""
    return np.broadcast_to(dx * dy, (rows, cols))


def horn_slope_deg(z_pad, dx, dy, rows, cols):
    """Slope (degrees) per pixel by Horn (1981): 3x3 finite differences on the
    true per-row ground spacing. Used only as a terrain descriptor; surface
    area comes from jenness_surface_area.

    z_pad : elevation, shape (rows+2, cols+2) -- 1-pixel halo on every side.
    dx, dy : per-row ground spacing, shape (rows, 1).
    """
    z = lambda r, c: z_pad[r:r + rows, c:c + cols]  # noqa: E731
    dzdx = ((z(0, 2) + 2 * z(1, 2) + z(2, 2)) - (z(0, 0) + 2 * z(1, 0) + z(2, 0))) / (8.0 * dx)
    dzdy = ((z(2, 0) + 2 * z(2, 1) + z(2, 2)) - (z(0, 0) + 2 * z(0, 1) + z(0, 2))) / (8.0 * dy)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def jenness_surface_area(z_pad, dx, dy, rows, cols):
    """Jenness (2004) triangulated surface area s_p (m^2) per pixel.

    Each pixel is split into 8 triangles spanned by its centre, its 4 corners
    (mean of the surrounding 2x2 pixel centres) and its 4 edge midpoints (mean
    of the two bounding corners).

    z_pad : elevation, shape (rows+2, cols+2) -- 1-pixel halo on every side.
    dx, dy : per-row ground spacing, shape (rows, 1).
    """
    z_c = z_pad[1:rows + 1, 1:cols + 1]

    nw = (z_pad[0:rows, 0:cols] + z_pad[0:rows, 1:cols + 1] +
          z_pad[1:rows + 1, 0:cols] + z_pad[1:rows + 1, 1:cols + 1]) / 4.0
    ne = (z_pad[0:rows, 1:cols + 1] + z_pad[0:rows, 2:cols + 2] +
          z_pad[1:rows + 1, 1:cols + 1] + z_pad[1:rows + 1, 2:cols + 2]) / 4.0
    sw = (z_pad[1:rows + 1, 0:cols] + z_pad[1:rows + 1, 1:cols + 1] +
          z_pad[2:rows + 2, 0:cols] + z_pad[2:rows + 2, 1:cols + 1]) / 4.0
    se = (z_pad[1:rows + 1, 1:cols + 1] + z_pad[1:rows + 1, 2:cols + 2] +
          z_pad[2:rows + 2, 1:cols + 1] + z_pad[2:rows + 2, 2:cols + 2]) / 4.0

    z_pts = {
        "NW": nw, "NE": ne, "SW": sw, "SE": se,
        "midN": (nw + ne) / 2.0, "midS": (sw + se) / 2.0,
        "midE": (ne + se) / 2.0, "midW": (nw + sw) / 2.0,
    }
    hx, hy = dx / 2.0, dy / 2.0
    zero = np.zeros_like(hx)
    xy = {
        "NW": (-hx, hy), "NE": (hx, hy), "SW": (-hx, -hy), "SE": (hx, -hy),
        "midN": (zero, hy), "midS": (zero, -hy), "midE": (hx, zero), "midW": (-hx, zero),
    }
    triangles = [
        ("NW", "midN"), ("midN", "NE"), ("NE", "midE"), ("midE", "SE"),
        ("SE", "midS"), ("midS", "SW"), ("SW", "midW"), ("midW", "NW"),
    ]

    area = np.zeros((rows, cols), dtype=np.float64)
    for p1, p2 in triangles:
        x1, y1 = xy[p1]
        x2, y2 = xy[p2]
        dz1 = z_pts[p1] - z_c
        dz2 = z_pts[p2] - z_c
        cx = y1 * dz2 - dz1 * y2
        cy = dz1 * x2 - x1 * dz2
        cz = x1 * y2 - y1 * x2
        area += 0.5 * np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    return area


def void_in_neighbourhood(z_pad, rows, cols):
    """True where any elevation in the pixel's 3x3 neighbourhood is a void."""
    void = z_pad <= ELEVATION_VOID_THRESHOLD
    out = np.zeros((rows, cols), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= void[dr:dr + rows, dc:dc + cols]
    return out
