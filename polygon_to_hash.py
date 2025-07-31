from polygon_geohasher.polygon_geohasher import polygon_to_geohashes, geohashes_to_polygon
from shapely import geometry
from rdflib import Graph, Namespace

def polygon_to_geohash(polygon_coords: list, precision: int):
    """
    Convert a given list of lat, lng coordinate pairs into respective geohashes.
    Returns inner geohashes (everything within polygon) and outer geohashes (everything on edge of polygon).
    """
    
    polygon = geometry.Polygon(polygon_coords)
    inner_geohashes = polygon_to_geohashes(polygon, precision, inner=True)
    outer_geohashes = polygon_to_geohashes(polygon, precision, inner=False)

    return inner_geohashes, outer_geohashes

def parse_wkt_polygon(wkt_str: str):
    """
    Turn a string like
      "POLYGON((x1 y1, x2 y2, …, xn yn))"
    into a Python list of (x, y) tuples.
    """
    # remove the prefix/suffix
    if not wkt_str.startswith("POLYGON((") or not wkt_str.endswith("))"):
        raise ValueError(f"Unexpected WKT: {wkt_str!r}")
    inner = wkt_str[len("POLYGON(("):-2]

    # split into coordinate strings, then map to floats
    coords = []
    for pair in inner.split(","):
        x_str, y_str = pair.strip().split()
        coords.append((float(x_str), float(y_str)))
    return coords

def extract_county_geohash_data(ttl_file: str, precision: int):
    """
    Read given ontology ttl file, extract all entries with a predicate "GEOSPARQL:asWKT" and convert polygon string
    to polygon list, then to inner and outer geohashes.
    """
    g = Graph()
    g.parse(ttl_file, format="turtle")

    # 2. Define namespaces
    GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
    GOO = Namespace("https://ucf-henat.github.io/GeoOutageOnto/#")
    GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")

    # 3. (Optional) Bind prefixes for nicer serialization
    g.bind("geo", GEO)
    g.bind("goo", GOO)
    g.bind("geosparql", GEOSPARQL)

    # 4. Iterate through each SpatialThing that has a latitude
    for subj in g.subjects(predicate=GEOSPARQL.asWKT):
        county_polygon = g.value(subject=subj, predicate=GEOSPARQL.asWKT)
        county_polygon_list = parse_wkt_polygon(county_polygon)
        inner_geohashes, outer_geohashes = polygon_to_geohash(county_polygon_list, precision)

        print(subj)
        print(f"Inner Geohashes: {inner_geohashes}")
        print(f"Outer Geohashes: {outer_geohashes}\n")

if __name__ == "__main__":
    # polygon_coords = [(-99.1795917, 19.432134), (-99.1656847, 19.429034),
    #                   (-99.1776492, 19.414236), (-99.1795917, 19.432134)]
    # polygon_to_geohash(polygon_coords, 4)

    ttl_file = "GeoOutageOnto_2_0_Beta.ttl"
    extract_county_geohash_data(ttl_file, 3)