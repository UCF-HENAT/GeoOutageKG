from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from pprint import pprint

# NHC Data Archive lists the latest “best track” HURDAT2 files.
DEFAULT_NHC_DATA_PAGE = "https://www.nhc.noaa.gov/data/"
DEFAULT_ATLANTIC_TXT = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"
DEFAULT_NEPAC_TXT = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2024-031725.txt"

# Common HURDAT2 “status” codes (aka intensity/phase in many workflows)
KNOWN_STATUS = {"TD", "TS", "HU", "EX", "SD", "SS", "LO", "WV", "DB"}


@dataclass(frozen=True)
class HurdatRecord:
    date_yyyymmdd: str
    time_hhmm: str
    record_identifier: str
    status: str
    lat: float
    lon: float
    max_wind_kt: int
    min_pressure_mb: int
    wind_radii_34kt: Dict[str, int]
    wind_radii_50kt: Dict[str, int]
    wind_radii_64kt: Dict[str, int]
    max_wind_radius: int


@dataclass(frozen=True)
class HurdatStorm:
    storm_id: str
    name: str
    n_records: int
    records: List[HurdatRecord]


def _parse_latlon(value: str) -> float:
    """Convert e.g. '16.5N' / '78.9W' into signed float degrees."""
    v = value.strip().upper()
    if not v:
        return float("nan")
    hemi = v[-1]
    num = float(v[:-1])
    if hemi in ("S", "W"):
        num = -num
    return num


def _to_int(x: str, default: int = -999) -> int:
    x = (x or "").strip()
    if x == "":
        return default
    try:
        return int(x)
    except ValueError:
        return default


def _storm_basin_from_id(storm_id: str) -> str:
    """
    Basin from storm_id prefix:
      AL = Atlantic
      EP/CP = East/Central Pacific (HURDAT2 "nepac" file)
    """
    sid = (storm_id or "").strip().upper()
    if sid.startswith("AL"):
        return "ATL"
    if sid.startswith(("EP", "CP")):
        return "EPAC"
    return "UNKNOWN"


def _normalize_basin_arg(basin: Optional[str]) -> Optional[str]:
    """
    Normalize user basin inputs to internal basin labels:
      - "ATL" for Atlantic
      - "EPAC" for East/Central Pacific (HURDAT2 nepac file)
    Accepted inputs:
      Atlantic: AL, ATL, ATLANTIC
      Pacific:  EP, CP, EPAC, NEPAC, PAC, PACIFIC
    """
    if basin is None:
        return None
    b = str(basin).strip().upper()
    if not b:
        return None

    atl = {"AL", "ATL", "ATLANTIC"}
    epac = {"EP", "CP", "EPAC", "NEPAC", "PAC", "PACIFIC"}

    if b in atl:
        return "ATL"
    if b in epac:
        return "EPAC"

    # Unknown -> don't filter (search both), but keep deterministic behavior
    return None


def _pick_latest(found_txt: List[str], *, want_nepac: bool) -> Optional[str]:
    """
    Pick newest-looking HURDAT2 filename for a basin from a list of filenames.
    Heuristic: filenames end with 6-digit stamp; we sort by that stamp descending.
    """
    def stamp(fname: str) -> str:
        m = re.search(r"(\d{6})\.txt$", fname)
        return m.group(1) if m else "000000"

    def is_nepac(fname: str) -> bool:
        return "nepac" in fname.lower()

    candidates = [f for f in found_txt if is_nepac(f)] if want_nepac else [f for f in found_txt if not is_nepac(f)]
    candidates = sorted(candidates, key=stamp, reverse=True)
    return candidates[0] if candidates else None


def _resolve_data_urls(base_url: str) -> List[Tuple[str, str]]:
    """
    Resolve URLs for BOTH basins:
      returns [("ATL", <atl_txt_url>), ("EPAC", <nepac_txt_url>)]
    Strategy:
      - If base_url points to a .txt, use it for its basin and resolve the other basin via scraping/fallback.
      - Else scrape provided page + NHC data page for latest HURDAT2 filenames.
      - Final fallback: pinned DEFAULT_* URLs.
    """
    b = (base_url or "").strip()

    # Pages to scrape (best effort)
    pages = []
    if b and not b.lower().endswith(".txt"):
        pages.append(b)
    if DEFAULT_NHC_DATA_PAGE not in pages:
        pages.append(DEFAULT_NHC_DATA_PAGE)

    found_txt: List[str] = []
    for page in pages:
        try:
            r = requests.get(page, timeout=20)
            r.raise_for_status()
            html = r.text
        except Exception:
            continue
        found_txt += re.findall(r"hurdat2-[\w\-]+?\d{6}\.txt", html, flags=re.IGNORECASE)

    found_txt = sorted(set(found_txt))

    # If user passed a .txt directly, use it for its basin
    direct_txt = b if b.lower().endswith(".txt") else None

    # Determine latest candidates from scrape
    atl_fname = _pick_latest(found_txt, want_nepac=False)
    nep_fname = _pick_latest(found_txt, want_nepac=True)

    def to_full_url(fname: Optional[str]) -> Optional[str]:
        if not fname:
            return None
        if fname.lower().startswith("http"):
            return fname
        return "https://www.nhc.noaa.gov/data/hurdat/" + fname

    atl_url = to_full_url(atl_fname) or DEFAULT_ATLANTIC_TXT
    nep_url = to_full_url(nep_fname) or DEFAULT_NEPAC_TXT

    if direct_txt:
        # Override whichever basin this direct file belongs to
        if "nepac" in direct_txt.lower():
            nep_url = direct_txt
        else:
            atl_url = direct_txt

    return [("ATL", atl_url), ("EPAC", nep_url)]


def _download_to_cache(url: str) -> Path:
    """Download the HURDAT2 .txt into a local cache dir and return the file path."""
    cache_dir = Path(os.getenv("SPRINT_HURDAT_CACHE_DIR", Path.home() / ".cache" / "sprint" / "hurdat"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = url.split("/")[-1] or "hurdat2.txt"
    path = cache_dir / fname

    # Simple cache: if present and non-empty, reuse
    if path.exists() and path.stat().st_size > 0:
        return path

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 128):
            if chunk:
                f.write(chunk)
    return path


# -------------------------
# Date/time query helpers
# -------------------------

def _normalize_date_yyyymmdd(s: Optional[str]) -> Optional[str]:
    """
    Normalize various date inputs to 'YYYYMMDD'.
    Accepts: YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD, etc.
    """
    if s is None:
        return None
    raw = str(s).strip()
    if not raw:
        return None

    digits = re.sub(r"\D+", "", raw)
    # If user passed YYYYMMDDHHMM in the "date" field, keep first 8 here.
    if len(digits) >= 8:
        return digits[:8]
    return None


def _normalize_time_hhmm(s: Optional[str]) -> Optional[str]:
    """
    Normalize various time inputs to 'HHMM' (24h).
    Accepts: HHMM, HMM, HH:MM, H:MM, '6' -> 0600, '600' -> 0600, etc.
    """
    if s is None:
        return None
    raw = str(s).strip()
    if not raw:
        return None

    digits = re.sub(r"\D+", "", raw)
    if not digits:
        return None

    # If 1-2 digits, interpret as hour.
    if len(digits) <= 2:
        try:
            hh = int(digits)
        except ValueError:
            return None
        if 0 <= hh <= 23:
            return f"{hh:02d}00"
        return None

    # "600" -> "0600"
    if len(digits) == 3:
        digits = "0" + digits

    if len(digits) >= 4:
        hhmm = digits[:4]
        try:
            hh = int(hhmm[:2])
            mm = int(hhmm[2:])
        except ValueError:
            return None
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}{mm:02d}"

    return None


def _coerce_query_date_time(date: Optional[str], time: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Coerce (date, time) into (YYYYMMDD, HHMM).

    Backwards-compatible nicety:
      - If `date` contains an embedded time (e.g., "2024-10-09 06:00" or "202410090600"),
        we will extract both even if `time` is None.
    """
    if date is None and time is None:
        return (None, None)

    d_raw = (str(date).strip() if date is not None else "")
    t_raw = (str(time).strip() if time is not None else "")

    # If time provided explicitly, normalize independently.
    q_time = _normalize_time_hhmm(t_raw) if t_raw else None

    # If `date` looks like it includes a time, try parsing it as datetime first.
    if d_raw:
        d_try = d_raw.strip().replace("z", "Z")
        if d_try.endswith("Z"):
            d_try = d_try[:-1].strip()

        fmts = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H%M",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d %H%M",
            "%Y%m%d%H%M",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%dT%H%M",
            "%Y%m%dT%H%M",
        ]
        for fmt in fmts:
            try:
                dt = datetime.strptime(d_try, fmt)
                q_date = dt.strftime("%Y%m%d")
                if q_time is None:
                    q_time = dt.strftime("%H%M")
                return (q_date, q_time)
            except Exception:
                pass

    # Fallback: date-only normalization
    q_date = _normalize_date_yyyymmdd(d_raw) if d_raw else None

    # If date was something like YYYYMMDDHHMM and time wasn't provided, extract HHMM
    if q_time is None and d_raw:
        digits = re.sub(r"\D+", "", d_raw)
        if len(digits) >= 12:  # YYYYMMDDHHMM
            q_time = _normalize_time_hhmm(digits[8:12])

    return (q_date, q_time)


# -------------------------
# Geohash helpers (no deps)
# -------------------------

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEOHASH_ALLOWED = set(_GEOHASH_BASE32)


def _normalize_geohash(s: Optional[str]) -> Optional[str]:
    """
    Normalize a geohash string (any length).
    Returns lowercase geohash if valid; else None.
    """
    if s is None:
        return None
    g = str(s).strip().lower()
    if not g:
        return None
    g = re.sub(r"\s+", "", g)
    if any(ch not in _GEOHASH_ALLOWED for ch in g):
        return None
    return g


def _geohash_encode(lat: float, lon: float, precision: int) -> str:
    """
    Encode (lat, lon) into a geohash string of given precision.
    Standard base32 geohash encoding; no external deps.
    """
    if precision <= 0:
        return ""

    # Clamp to valid ranges (keeps behavior sane on edge cases)
    lat = max(-90.0, min(90.0, float(lat)))
    lon = max(-180.0, min(180.0, float(lon)))

    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]

    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True

    out: List[str] = []
    while len(out) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2.0
            if lon >= mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2.0
            if lat >= mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid

        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_GEOHASH_BASE32[ch])
            bit = 0
            ch = 0

    return "".join(out)


def _iter_storms(lines: Iterable[str]) -> Iterable[HurdatStorm]:
    """
    Stream-parse HURDAT2 storms.
    Header line: ID, NAME, N,
    Then N record lines.
    """
    it = iter(lines)
    for raw in it:
        line = raw.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue

        storm_id = parts[0].upper()
        name = parts[1].upper()
        n_records = _to_int(parts[2], default=0)

        records: List[HurdatRecord] = []
        for _ in range(max(n_records, 0)):
            rec_line = next(it).strip()
            rec_parts = [p.strip() for p in rec_line.split(",")]

            # Minimum expected columns per format example:
            # date, time, record_id, status, lat, lon, wind, pressure,
            # wind_radii_34kt (4), wind_radii_50kt (4), wind_radii_64kt (4), max_wind_radius
            # Radii quadrants are [NE, SE, SW, NW] in nautical miles.

            date_yyyymmdd = rec_parts[0] if len(rec_parts) > 0 else ""
            time_hhmm = (rec_parts[1] if len(rec_parts) > 1 else "").strip()
            if time_hhmm.isdigit():
                time_hhmm = time_hhmm.zfill(4)

            record_identifier = rec_parts[2] if len(rec_parts) > 2 else ""
            status = rec_parts[3].upper() if len(rec_parts) > 3 else ""

            lat = _parse_latlon(rec_parts[4]) if len(rec_parts) > 4 else float("nan")
            lon = _parse_latlon(rec_parts[5]) if len(rec_parts) > 5 else float("nan")
            wind = _to_int(rec_parts[6], default=-999) if len(rec_parts) > 6 else -999
            pres = _to_int(rec_parts[7], default=-999) if len(rec_parts) > 7 else -999

            # Radii blocks can be missing in older entries; pad safely.
            def safe_int_at(i: int) -> int:
                return _to_int(rec_parts[i], default=-999) if i < len(rec_parts) else -999

            wind_radii_34kt: Dict[str, int] = {
                "NE": safe_int_at(8),
                "SE": safe_int_at(9),
                "SW": safe_int_at(10),
                "NW": safe_int_at(11),
            }
            wind_radii_50kt: Dict[str, int] = {
                "NE": safe_int_at(12),
                "SE": safe_int_at(13),
                "SW": safe_int_at(14),
                "NW": safe_int_at(15),
            }
            wind_radii_64kt: Dict[str, int] = {
                "NE": safe_int_at(16),
                "SE": safe_int_at(17),
                "SW": safe_int_at(18),
                "NW": safe_int_at(19),
            }

            # Safely index max_wind_radius
            max_wind_radius = _to_int(rec_parts[20], default=-999) if len(rec_parts) > 20 else -999

            records.append(
                HurdatRecord(
                    date_yyyymmdd=date_yyyymmdd,
                    time_hhmm=time_hhmm,
                    record_identifier=record_identifier,
                    status=status,
                    lat=lat,
                    lon=lon,
                    max_wind_kt=wind,
                    min_pressure_mb=pres,
                    wind_radii_34kt=wind_radii_34kt,
                    wind_radii_50kt=wind_radii_50kt,
                    wind_radii_64kt=wind_radii_64kt,
                    max_wind_radius=max_wind_radius,
                )
            )

        yield HurdatStorm(storm_id=storm_id, name=name, n_records=n_records, records=records)


def _storm_summary(st: HurdatStorm, *, include_records: bool, basin: str) -> Dict[str, Any]:
    year = _to_int(st.storm_id[-4:], default=-1)

    max_wind = max((r.max_wind_kt for r in st.records if r.max_wind_kt != -999), default=-999)
    min_pres = min((r.min_pressure_mb for r in st.records if r.min_pressure_mb != -999), default=-999)

    status_counts: Dict[str, int] = {}
    for r in st.records:
        if r.status:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

    def dt(rec: HurdatRecord) -> Optional[str]:
        try:
            d = datetime.strptime(rec.date_yyyymmdd + (rec.time_hhmm or "").zfill(4), "%Y%m%d%H%M")
            return d.isoformat() + "Z"
        except Exception:
            return None

    start_dt = dt(st.records[0]) if st.records else None
    end_dt = dt(st.records[-1]) if st.records else None

    out: Dict[str, Any] = {
        "id": st.storm_id,
        "name": st.name,
        "basin": basin,
        "season": year,
        "n_records": st.n_records,
        "start_time": start_dt,
        "end_time": end_dt,
        "max_wind_kt": max_wind,
        "min_pressure_mb": min_pres,
        "status_counts": status_counts,
    }

    if include_records:
        out["records"] = [
            {
                "date": r.date_yyyymmdd,
                "time": (r.time_hhmm or "").zfill(4),
                "record_identifier": r.record_identifier,
                "status": r.status,
                "lat": r.lat,
                "lon": r.lon,
                "max_wind_kt": r.max_wind_kt,
                "min_pressure_mb": r.min_pressure_mb,
                "wind_radii_34kt": r.wind_radii_34kt,
                "wind_radii_50kt": r.wind_radii_50kt,
                "wind_radii_64kt": r.wind_radii_64kt,
                "max_wind_radius": r.max_wind_radius,
            }
            for r in st.records
        ]

    return out


def search_hurdat(
    base_url: str = "https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html",
    cyclone_name: Optional[str] = None,
    cyclone_season: Optional[str] = None,
    cyclone_number: Optional[str] = None,
    cyclone_intensity: Optional[str] = None,
    cyclone_basin: Optional[str] = None,  # <-- basin filter
    date: Optional[str] = None,           # <-- date filter (YYYYMMDD / YYYY-MM-DD / may include time)
    time: Optional[str] = None,           # <-- time filter (HHMM / HH:MM / "6" -> 0600)
    geohash: Optional[str] = None,        # <-- geohash filter (any length; record must fall inside that cell)
    return_metadata: bool = True,
    max_rows: int = 25,
) -> Dict[str, Any]:
    """
    Search HURDAT2 by:
      - cyclone_name (case-insensitive substring match)
      - cyclone_season (year)
      - cyclone_number (e.g., AL092021, EP182015)
      - cyclone_intensity/status (e.g., HU, TS, TD, EX, ...)
      - cyclone_basin (e.g., AL for Atlantic, EP/CP for East/Central Pacific)
      - date (YYYYMMDD or YYYY-MM-DD; may also include time like 'YYYY-MM-DD 06:00')
      - time (HHMM / HH:MM / '6' => 0600; optional)
      - geohash (any length; matches storms with >=1 record inside that geohash cell)

    Record-level filtering (date/time/geohash) matches storms that contain at least one best-track record
    satisfying ALL specified record-level constraints.

    Searches BOTH ATL + EPAC unless cyclone_number/cyclone_basin implies a single basin.
    Returns first max_rows matching storms across basins.
    """
    q_name = (cyclone_name or "").strip().upper() or None
    q_year = _to_int(str(cyclone_season), default=-1) if cyclone_season is not None else None
    q_num = (cyclone_number or "").strip().upper() or None
    q_int = (cyclone_intensity or "").strip().upper() or None
    q_basin = _normalize_basin_arg(cyclone_basin)

    q_date, q_time = _coerce_query_date_time(date, time)
    q_geohash = _normalize_geohash(geohash)
    gh_prec = len(q_geohash) if q_geohash else 0

    # Basin filter precedence:
    # 1) explicit cyclone_basin
    # 2) inferred from cyclone_number
    # 3) None => search both
    basin_filter: Optional[str] = q_basin
    if basin_filter is None and q_num:
        inferred = _storm_basin_from_id(q_num)
        basin_filter = None if inferred == "UNKNOWN" else inferred

    urls = _resolve_data_urls(base_url)

    sources: Dict[str, Any] = {
        "dataset": "HURDAT2",
        "requested_basin": cyclone_basin,
        "normalized_basin": basin_filter,
        "resolved_data_urls": {},
        "cache_files": {},
        "known_status_codes": sorted(KNOWN_STATUS),
        "warnings": [],
    }

    # If user provided geohash but it is invalid, return 0 results with a clear warning.
    if geohash is not None and q_geohash is None:
        sources["warnings"].append(
            f"Invalid geohash {geohash!r}. Expected only characters in: {_GEOHASH_BASE32}"
        )
        return {
            "query": {
                "cyclone_name": cyclone_name,
                "cyclone_season": cyclone_season,
                "cyclone_number": cyclone_number,
                "cyclone_intensity": cyclone_intensity,
                "cyclone_basin": cyclone_basin,
                "date": date,
                "time": time,
                "geohash": geohash,
                "normalized_date": q_date,
                "normalized_time": q_time,
                "normalized_geohash": q_geohash,
                "max_rows": max_rows,
            },
            "count": 0,
            "items": [],
            "source": sources,
        }

    items: List[Dict[str, Any]] = []
    count = 0

    for basin, url in urls:
        if basin_filter and basin != basin_filter:
            continue

        local_path = _download_to_cache(url)
        sources["resolved_data_urls"][basin] = url
        sources["cache_files"][basin] = str(local_path)

        with local_path.open("r", encoding="utf-8", errors="replace") as f:
            for storm in _iter_storms(f):
                # Apply storm-level filters
                if q_num and storm.storm_id != q_num:
                    continue

                if q_year is not None and q_year != -1:
                    year = _to_int(storm.storm_id[-4:], default=-1)
                    if year != q_year:
                        continue

                if q_name and q_name not in storm.name:
                    continue

                if q_int:
                    if not any(r.status == q_int for r in storm.records):
                        continue

                # Record-level filters (date/time/geohash):
                matched_records: Optional[List[HurdatRecord]] = None
                if q_date or q_time or q_geohash:
                    matched_records = []
                    for r in storm.records:
                        if q_date and r.date_yyyymmdd != q_date:
                            continue
                        if q_time and (r.time_hhmm or "").zfill(4) != q_time:
                            continue
                        if q_geohash:
                            if not (math.isfinite(r.lat) and math.isfinite(r.lon)):
                                continue
                            if _geohash_encode(r.lat, r.lon, precision=gh_prec) != q_geohash:
                                continue
                        matched_records.append(r)
                    if not matched_records:
                        continue

                out = _storm_summary(storm, include_records=bool(return_metadata), basin=basin)

                # If record-level filtering was used, include the matching record(s) explicitly
                if (q_date or q_time or q_geohash) and matched_records is not None:
                    out["matched_records"] = [
                        {
                            "date": r.date_yyyymmdd,
                            "time": (r.time_hhmm or "").zfill(4),
                            "record_identifier": r.record_identifier,
                            "status": r.status,
                            "lat": r.lat,
                            "lon": r.lon,
                            **({"geohash": _geohash_encode(r.lat, r.lon, precision=gh_prec)} if q_geohash else {}),
                            "max_wind_kt": r.max_wind_kt,
                            "min_pressure_mb": r.min_pressure_mb,
                            "wind_radii_34kt": r.wind_radii_34kt,
                            "wind_radii_50kt": r.wind_radii_50kt,
                            "wind_radii_64kt": r.wind_radii_64kt,
                            "max_wind_radius": r.max_wind_radius,
                        }
                        for r in matched_records
                    ]

                items.append(out)
                count += 1
                if count >= int(max_rows or 25):
                    break

        if count >= int(max_rows or 25):
            break

    return {
        "query": {
            "cyclone_name": cyclone_name,
            "cyclone_season": cyclone_season,
            "cyclone_number": cyclone_number,
            "cyclone_intensity": cyclone_intensity,
            "cyclone_basin": cyclone_basin,
            "date": date,
            "time": time,
            "geohash": geohash,
            "normalized_date": q_date,
            "normalized_time": q_time,
            "normalized_geohash": q_geohash,
            "max_rows": max_rows,
        },
        "count": count,
        "items": items,
        "source": sources,
    }


if __name__ == "__main__":
    # Example 1: search by hurricane season
    hurdat_query1 = search_hurdat(
        cyclone_season="2005",
        cyclone_basin="AL",
        return_metadata=False,
        max_rows=50
    )
    pprint(hurdat_query1)

    # Example 2: date-only search (any record on that date)
    hurdat_query2 = search_hurdat(
        base_url="https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html",
        cyclone_basin="AL",
        date="2005-08-29",
        return_metadata=False,
        max_rows=10,
    )
    pprint(hurdat_query2)

    # Example 3: exact date+time record match
    hurdat_query3 = search_hurdat(
        cyclone_basin="AL",
        date="2024-10-08",
        time="12:00",
        return_metadata=False,
        max_rows=10,
    )
    pprint(hurdat_query3)

    # Example 4: geohash search (any length; storms with >=1 record in that cell)
    hurdat_query4 = search_hurdat(
        cyclone_basin="AL",
        geohash="dh",          # broader cell
        date="2022-09-29",
        return_metadata=False,
        max_rows=10,
    )
    pprint(hurdat_query4)

    # Example 5: narrower geohash + date (record must satisfy both)
    hurdat_query5 = search_hurdat(
        cyclone_basin="AL",
        geohash="dhtww",
        date="2005-08-29",
        return_metadata=False,
        max_rows=10,
    )
    pprint(hurdat_query5)