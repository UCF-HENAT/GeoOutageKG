#!/usr/bin/env python3
"""
find_osf_outage_map_url.py

Example:
  python find_osf_outage_map_url.py --date 2018-09-11
  python find_osf_outage_map_url.py --date 2018_09_11 --county alachua
  python find_osf_outage_map_url.py --node qvd8b --base-dir outage_maps --county alachua --date 2018-09-11 --mode both
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl

import requests


OSF_API_BASE = "https://api.osf.io/v2"


def normalize_date_token(date_str: str) -> str:
    """
    Accepts YYYY-MM-DD or YYYY_MM_DD (and a few close cousins) and returns YYYY_MM_DD.
    """
    s = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y_%m_%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y_%m_%d")
        except ValueError:
            pass
    # If user already passed something like 2018_09_11_extra, keep it as-is
    # but still try to be helpful.
    return s.replace("-", "_").replace("/", "_")


def add_query_params(url: str, params: Dict[str, str]) -> str:
    parts = urlsplit(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q.update({k: v for k, v in params.items() if v is not None})
    new_query = urlencode(q, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def extract_related_href(item: Dict[str, Any], relationship: str) -> Optional[str]:
    """
    Tries to pull relationships[relationship].links.related.href from JSON:API-like payloads.
    """
    rel = (item.get("relationships") or {}).get(relationship) or {}
    links = rel.get("links") or {}
    related = links.get("related")
    if isinstance(related, str):
        return related
    if isinstance(related, dict):
        return related.get("href") or related.get("url")
    return None


def osf_get_json(url: str, *, timeout: float = 30.0) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def iter_collection(url: str, *, page_size: int = 200, extra_params: Optional[Dict[str, str]] = None) -> Iterable[Dict[str, Any]]:
    """
    Iterates through OSF API collections, following JSON:API pagination if present.
    """
    params = {"page[size]": str(page_size)}
    if extra_params:
        params.update(extra_params)

    next_url = add_query_params(url, params)

    while next_url:
        payload = osf_get_json(next_url)
        for obj in payload.get("data") or []:
            if isinstance(obj, dict):
                yield obj

        links = payload.get("links") or {}
        next_link = links.get("next")
        next_url = next_link if isinstance(next_link, str) else None


def find_child_by_name(
    list_url: str,
    name: str,
    *,
    want_kind: Optional[str] = None,
    page_size: int = 200,
) -> Optional[Dict[str, Any]]:
    """
    Finds an entry in a folder listing by exact name match. Tries server-side filter first,
    then falls back to scanning.
    """
    # Try filter[name]=... if supported
    try:
        for obj in iter_collection(list_url, page_size=page_size, extra_params={"filter[name]": name}):
            attrs = obj.get("attributes") or {}
            if (attrs.get("name") == name) and (want_kind is None or attrs.get("kind") == want_kind):
                return obj
    except requests.HTTPError:
        pass  # fall back to scanning without filter

    # Full scan fallback
    for obj in iter_collection(list_url, page_size=page_size):
        attrs = obj.get("attributes") or {}
        if (attrs.get("name") == name) and (want_kind is None or attrs.get("kind") == want_kind):
            return obj
    return None


def traverse_folders(node_id: str, provider: str, folder_parts: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Walks from node root -> folder_parts, returning final listing URL and the breadcrumb items.
    """
    current_list_url = f"{OSF_API_BASE}/nodes/{node_id}/files/{provider}/"
    breadcrumbs: List[Dict[str, Any]] = []

    for part in folder_parts:
        folder_obj = find_child_by_name(current_list_url, part, want_kind="folder")
        if not folder_obj:
            raise FileNotFoundError(f"Folder '{part}' not found under listing: {current_list_url}")

        breadcrumbs.append(folder_obj)

        next_list_url = extract_related_href(folder_obj, "files")
        if not next_list_url:
            # Some payloads may put a direct "files" listing link under attributes/links; try a few fallbacks
            links = folder_obj.get("links") or {}
            next_list_url = links.get("related") or links.get("self")

        if not next_list_url:
            raise RuntimeError(f"Could not determine child listing URL for folder '{part}'")

        current_list_url = next_list_url

    return current_list_url, breadcrumbs


def pick_urls(item: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Pulls useful URLs from a file/folder JSON object.
    """
    links = item.get("links") or {}
    attrs = item.get("attributes") or {}

    out: Dict[str, Optional[str]] = {
        "name": attrs.get("name"),
        "kind": attrs.get("kind"),
        "path": attrs.get("path") or attrs.get("materialized_path"),
        "download": links.get("download"),
        "html": links.get("html"),
        "self": links.get("self"),
    }

    # If there's a GUID, provide a classic OSF download link too (handy for curl/wget).
    guid = attrs.get("guid")
    if guid and isinstance(guid, str):
        out["osf_guid_download"] = f"https://osf.io/{guid}/?action=download"
        out["osf_guid_page"] = f"https://osf.io/{guid}/"
    else:
        out["osf_guid_download"] = None
        out["osf_guid_page"] = None

    return out


def find_matches_in_folder(list_url: str, date_token: str) -> List[Dict[str, Any]]:
    """
    Finds file/folder entries whose name matches the date_token (exact or prefix).
    """
    matches: List[Dict[str, Any]] = []

    # Try server-side exact-name filter first
    try:
        for obj in iter_collection(list_url, extra_params={"filter[name]": date_token}):
            matches.append(obj)
        if matches:
            return matches
    except requests.HTTPError:
        pass

    # Scan and match by exact or prefix (to catch extensions like .png, .tif, etc.)
    for obj in iter_collection(list_url):
        attrs = obj.get("attributes") or {}
        name = attrs.get("name") or ""
        if name == date_token or name.startswith(date_token):
            matches.append(obj)

    return matches


def main() -> int:
    p = argparse.ArgumentParser(description="Find an OSF file URL by folder path + date token.")
    p.add_argument("--node", default="qvd8b", help="OSF node id (default: qvd8b)")
    p.add_argument("--provider", default="osfstorage", help="Storage provider (default: osfstorage)")
    p.add_argument("--base-dir", default="outage_maps", help="Top-level folder (default: outage_maps)")
    p.add_argument("--county", default="alachua", help="County folder (default: alachua)")
    p.add_argument("--date", required=True, help="Date like 2018-09-11 or 2018_09_11")
    p.add_argument(
        "--mode",
        choices=("download", "html", "both"),
        default="both",
        help="Which URL(s) to print (default: both)",
    )

    args = p.parse_args()
    date_token = normalize_date_token(args.date)

    folder_parts = [args.base_dir, args.county]

    try:
        county_list_url, crumbs = traverse_folders(args.node, args.provider, folder_parts)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    matches = find_matches_in_folder(county_list_url, date_token)

    if not matches:
        # Optional: if the date token might be a folder, try to find that folder and report its URL.
        print(
            f"No matches found for date token '{date_token}' under '{'/'.join(folder_parts)}'.",
            file=sys.stderr,
        )
        return 1

    # Prefer files over folders if both exist
    def sort_key(obj: Dict[str, Any]) -> Tuple[int, str]:
        kind = (obj.get("attributes") or {}).get("kind")
        # files first (0), folders second (1)
        return (0 if kind == "file" else 1, (obj.get("attributes") or {}).get("name") or "")

    matches.sort(key=sort_key)

    # Print results
    for obj in matches:
        info = pick_urls(obj)

        if args.mode == "download":
            print(info.get("download") or info.get("osf_guid_download") or info.get("html") or info.get("self"))
        elif args.mode == "html":
            print(info.get("html") or info.get("self") or info.get("download") or info.get("osf_guid_page"))
        else:
            # both
            print("----")
            print(f"name: {info.get('name')}")
            print(f"kind: {info.get('kind')}")
            print(f"path: {info.get('path')}")
            print(f"download: {info.get('download')}")
            print(f"html: {info.get('html')}")
            if info.get("osf_guid_download"):
                print(f"osf_guid_download: {info.get('osf_guid_download')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
