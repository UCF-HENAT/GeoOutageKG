from __future__ import annotations

import math
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD, OWL
from hurdat_lookup import search_hurdat

# Define and bind namespaces
Goo    = Namespace("https://ucf-henat.github.io/GeoOutageOnto/#")
Gokg   = Namespace("http://example.org/resource#")
Schema = Namespace("http://schema.org/")
Geo    = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")

class HurdatKG:
    """
    Class for extracting HURDAT2 data using hurdat_lookup and serializing it 
    into a GeoOutage Ontology compliant Knowledge Graph.
    """
    def __init__(self, fmt: str = "ttl"):
        self.fmt = "turtle" if fmt == "ttl" else fmt

    def create_graph(self) -> Graph:
        g = Graph()
        g.bind("goo",    Goo)
        g.bind("gokg",   Gokg)
        g.bind("schema", Schema, override=True, replace=True)
        g.bind("rdfs",   RDFS)
        g.bind("rdf",    RDF)
        g.bind("xsd",    XSD)
        g.bind("geo",    Geo)

        return g

    def serialize_hurdat(self, out_path: str, max_rows: int = 100000):
        """
        Retrieves HURDAT data and writes HurricaneEvent and HurricaneTrackRecord 
        class instances to the specified output file.
        """
        g = self.create_graph()
        
        # Max rows set extremely high to pull the entire historic dataset
        hurdat_data = search_hurdat(return_metadata=True, max_rows=max_rows)

        for storm in hurdat_data.get("items", []):
            storm_id = storm["id"]
            name = storm["name"]
            basin = storm["basin"]
            start_time = storm["start_time"]  # Expected ISO 8601 string
            end_time = storm["end_time"]
            max_wind = storm["max_wind_kt"]
            min_pres = storm["min_pressure_mb"]

            # Define subject URI for the Hurricane Event
            storm_uri = Gokg[f"hurricaneevent.{storm_id}"]

            # Type and labels
            g.add((storm_uri, RDF.type, Goo.HurricaneEvent))
            g.add((storm_uri, RDFS.label, Literal(f"Hurricane {name} ({storm_id})", lang="en")))
            g.add((storm_uri, Schema.name, Literal(name, lang="en")))
            g.add((storm_uri, RDFS.comment, Literal(f"HURDAT2 data for {basin} storm {name} ({storm_id})", lang="en")))
            
            # Event level properties
            g.add((storm_uri, Goo.stormId, Literal(storm_id, datatype=XSD.string)))
            g.add((storm_uri, Goo.basin, Literal(basin, datatype=XSD.string)))

            if start_time:
                g.add((storm_uri, Goo.eventStartTime, Literal(start_time, datatype=XSD.dateTime)))
            if end_time:
                g.add((storm_uri, Goo.eventEndTime, Literal(end_time, datatype=XSD.dateTime)))
            if max_wind != -999:
                g.add((storm_uri, Goo.maxSustainedWindSpeed, Literal(max_wind, datatype=XSD.integer)))
            if min_pres != -999:
                g.add((storm_uri, Goo.minPressure, Literal(min_pres, datatype=XSD.integer)))

            # Iterate through the 6-hourly track records
            for rec in storm.get("records", []):
                date_str = rec["date"] # YYYYMMDD
                time_str = rec["time"] # HHMM
                
                # Format to conform to xsd:dateTime: YYYY-MM-DDTHH:MM:00Z
                dt_iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str[:2]}:{time_str[2:]}:00Z"
                
                # URI safe datetime string (avoid colons in URIs)
                dt_safe = dt_iso.replace(":", "-")

                track_uri = Gokg[f"hurricanetrack.{storm_id}.{dt_safe}"]

                # Track Type and Labels
                g.add((track_uri, RDF.type, Goo.HurricaneTrackRecord))
                g.add((track_uri, RDFS.label, Literal(f"{storm_id} Track at {dt_iso}", lang="en")))
                g.add((track_uri, Goo.recordDateTime, Literal(dt_iso, datatype=XSD.dateTime)))
                
                if rec["status"]:
                    g.add((track_uri, Goo.status, Literal(rec["status"], datatype=XSD.string)))

                # Coordinates
                lat = rec["lat"]
                lon = rec["lon"]
                if not math.isnan(lat):
                    g.add((track_uri, Geo.lat, Literal(lat, datatype=XSD.decimal)))
                if not math.isnan(lon):
                    g.add((track_uri, Geo.long, Literal(lon, datatype=XSD.decimal)))

                # Intensity
                if rec["max_wind_kt"] != -999:
                    g.add((track_uri, Goo.maxSustainedWindSpeed, Literal(rec["max_wind_kt"], datatype=XSD.integer)))
                if rec["min_pressure_mb"] != -999:
                    g.add((track_uri, Goo.minPressure, Literal(rec["min_pressure_mb"], datatype=XSD.integer)))

                # Object Property: Link the Event to the specific Track Record
                g.add((storm_uri, Goo.hasTrackRecord, track_uri))

        # Serialize to output file
        g.serialize(destination=out_path, format=self.fmt)

if __name__ == "__main__":
    kg_writer = HurdatKG(fmt="ttl")
    
    # Specify the output file name
    output_filename = "hurdat.ttl"
    
    print("Fetching HURDAT data and serializing to Knowledge Graph...")
    print("This may take a moment due to the size of the dataset.")
    
    # 100,000 will easily encapsulate both Atlantic and Pacific historic DBs
    kg_writer.serialize_hurdat(out_path=output_filename, max_rows=100000)
    
    print(f"Serialization complete! Data written to {output_filename}")