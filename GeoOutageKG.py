import os
import sys
import json
import pandas as pd
from datetime import datetime
from PIL import Image
import argparse

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

# Define and bind namespaces
Goo    = Namespace("https://ucf-henat.github.io/GeoOutageOnto/#")
Gokg   = Namespace("http://example.org/resource#")
Dbr    = Namespace("https://dbpedia.org/page/")
Ma     = Namespace("http://www.w3.org/ns/ma-ont#")
Schema = Namespace("http://schema.org/")

class GeoOutageKG:
    """
    Class for handling writing power outage data as GeoOutageKG class instances.
    Currently supports writing to one of three instance types: OutageRecord, NTLImage,
    and OutageMap. As more classes are added to GeoOutageOnto, more writing functionality
    will become available.
    """
    def __init__(self, ntl_dir: str, outage_map_dir: str, fips_json: str, fmt: str="turtle"):
        self.ntl_dir = ntl_dir
        self.outage_map_dir = outage_map_dir
        self.fips_json = fips_json
        self.fmt = fmt

    def create_graph(self):
        g = Graph()
        g.bind("goo",    Goo)
        g.bind("gokg",   Gokg)
        g.bind("dbr",    Dbr)
        g.bind("ma-ont", Ma)
        g.bind("schema", Schema, override=True, replace=True)
        g.bind("rdfs",   RDFS)
        g.bind("rdf",    RDF)
        g.bind("xsd",    XSD)

        return g

    def create_outage_records(self, year: int, root_dir: str, out_path: str):
        """
        Write OutageRecord class instances to given year file with given dataset.
        """
        g = self.create_graph()
        data_path = os.path.join(root_dir, f"eagle-i/24237376/florida_data/eaglei_outages_{year}.csv")
        df = pd.read_csv(data_path)

        for _, entry in df.iterrows():
            # Load metadata to add to instance from outage record entry
            county_name = entry["county"].replace(" ", "_")
            fips_code = entry["fips_code"]
            num_outages = entry.get("customers_out", entry.get("sum"))
            if pd.isna(num_outages):
                continue
            time = entry["run_start_time"]
            time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d_%H:%M:%S")
            recordDateTime = datetime.strptime(time, "%Y-%m-%d_%H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")
            recordDateTimettl = datetime.strptime(time, "%Y-%m-%d_%H:%M:%S").strftime("%Y-%m-%dT%H-%M-%SZ")
            date = datetime.strptime(time, "%Y-%m-%d_%H:%M:%S").strftime("%Y-%m-%d")
            date_file = datetime.strptime(date, "%Y-%m-%d").strftime("%Y_%m_%d")

            subj = Gokg[f"outagerecord.{fips_code}.{recordDateTimettl}"]

            # type and labels
            g.add((subj, RDF.type, Goo.OutageRecord))
            g.add((subj, RDFS.label,   Literal(f"{county_name} {recordDateTime}", lang="en")))
            g.add((subj, Schema.name,  Literal(f"{county_name} {recordDateTime}", lang="en")))
            g.add((subj, RDFS.comment, Literal(
                f"Eagle-I outage data for {county_name} County at time {recordDateTime}",
                lang="en")))
            
            # properties
            g.add((subj, Goo.representsCounty,
               URIRef(f"{Dbr}{county_name}_County%2C_Florida")))
            g.add((subj, Goo.recordDateTime,
                Literal(recordDateTime, datatype=XSD.dateTime)))
            g.add((subj, Goo.numberOfOutages,
                Literal(int(num_outages), datatype=XSD.integer)))
            
            # optional links
            ntl_path = os.path.join(self.ntl_dir, county_name.lower(), f"{date_file}.png")
            if os.path.exists(ntl_path):
                g.add((subj, Goo.hasNTLImage,
                    Gokg[f"ntlimage.{fips_code}.{date}"]))

            map_path = os.path.join(self.outage_map_dir, county_name.lower(), f"{date_file}.png")
            if os.path.exists(map_path):
                g.add((subj, Goo.hasOutageMap,
                    Gokg[f"outagemap.{fips_code}.{date}"]))
                
        # serialize
        g.serialize(destination=out_path, format=self.fmt)

    def create_ntl_images(self, out_path: str):
        """
        Write NTLImage class instances to given file with given dataset.
        """
        g = self.create_graph()
        with open(self.fips_json, 'r') as f:
            fips = json.load(f)

        for dirpath, _, files in os.walk(self.ntl_dir):
            print(dirpath)
            county_name = os.path.basename(dirpath).title().replace(" ", "_")
            for file in files:
                date = file.split('.')[0]
                iso_date = datetime.strptime(date, "%Y_%m_%d").strftime("%Y-%m-%d")
                fips_code = fips[county_name]

                subj = Gokg[f"ntlimage.{fips_code}.{iso_date}"]

                # type and labels
                g.add((subj, RDF.type, Goo.NTLImage))
                g.add((subj, RDFS.label,
                   Literal(f"{county_name} {iso_date}", lang="en")))
                g.add((subj, Schema.name,
                    Literal(f"{county_name} {iso_date}", lang="en")))
                g.add((subj, RDFS.comment,
                    Literal(f"Black Marble NTL Image for {county_name} County at date {iso_date}", lang="en")))
                g.add((subj, Goo.representsCounty,
                    URIRef(f"{Dbr}{county_name}_County%2C_Florida")))
                
                # dimensions
                with Image.open(os.path.join(dirpath, file)) as img:
                    width, height = img.size
                g.add((subj, Ma.frameWidth, Literal(width, datatype=XSD.integer)))
                g.add((subj, Ma.frameHeight, Literal(height, datatype=XSD.integer)))
                g.add((subj, Ma.date,        Literal(iso_date, datatype=XSD.date)))

                # locators
                g.add((subj, Ma.locator,
                   URIRef(f"http://127.0.0.1:5000/ntlimage/{county_name.lower()}/{date}.png")))
                g.add((subj, Ma.locator,
                    URIRef(f"http://127.0.0.1:5000/ntlimage/{county_name.lower()}/{date}.pkl")))
                
                # optional outage map link
                map_png = os.path.join(self.outage_map_dir, county_name.lower(), file)
                if os.path.exists(map_png):
                    g.add((subj, Goo.hasOutageMap, Gokg[f"outagemap.{fips_code}.{iso_date}"]))
        
        # serialize
        g.serialize(destination=out_path, format=self.fmt)

    def create_outage_maps(self, out_path: str):
        """
        Write OutageMap class instances to given file with given dataset.
        """
        g = self.create_graph()
        with open(self.fips_json, 'r') as f:
                    fips = json.load(f)

        for dirpath, _, files in os.walk(self.outage_map_dir):
            print(dirpath)
            county_name = os.path.basename(dirpath).title().replace(" ", "_")
            for file in files:
                date = file.split('.')[0]
                iso_date = datetime.strptime(date, "%Y_%m_%d").strftime("%Y-%m-%d")
                fips_code = fips[county_name]

                subj = Gokg[f"outagemap.{fips_code}.{iso_date}"]

                # type and labels
                g.add((subj, RDF.type, Goo.OutageMap))
                g.add((subj, RDFS.label,
                   Literal(f"{county_name} {iso_date}", lang="en")))
                g.add((subj, Schema.name,
                    Literal(f"{county_name} {iso_date}", lang="en")))
                g.add((subj, RDFS.comment,
                    Literal(f"Outage Map for {county_name} County at date {iso_date}", lang="en")))
                g.add((subj, Goo.representsCounty,
                    URIRef(f"{Dbr}{county_name}_County%2C_Florida")))
                
                # dimensions
                with Image.open(os.path.join(dirpath, file)) as img:
                    width, height = img.size
                g.add((subj, Ma.frameWidth,  Literal(width,  datatype=XSD.integer)))
                g.add((subj, Ma.frameHeight, Literal(height, datatype=XSD.integer)))
                g.add((subj, Ma.date,        Literal(iso_date, datatype=XSD.date)))

                # locator
                g.add((subj, Ma.locator,
                   URIRef(f"http://127.0.0.1:5000/outagemap/{county_name.lower()}/{date}.png")))
                
        g.serialize(destination=out_path, format=self.fmt)


# Example usage
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--fmt", type=str, default="ttl")
    args = p.parse_args()

    root_dir = args.root_dir
    fmt = args.fmt
    if fmt == "turtle":
        fmt = "ttl"

    ntl_dir        = os.path.join(root_dir, "VNP46A2_county_imgs/")
    outage_map_dir = os.path.join(root_dir, "outage_maps/")
    fips_json      = os.path.join(root_dir, "fips_codes.json")

    # Create GeoOutageKG instance
    gokg_write = GeoOutageKG(ntl_dir, outage_map_dir, fips_json, fmt=fmt)

    # Write TTL
    for year in range(2014, 2025):
        gokg_write.create_outage_records(year, root_dir, out_path=f"outagerecord_{year}.{fmt}")
        print(f"Serialized Outage Records for year {year}.")
        
    gokg_write.create_outage_maps(out_path=f"outagemap.{fmt}")
    print(f"Serialized Outage Maps.")

    gokg_write.create_ntl_images(out_path=f"ntlimage.{fmt}")
    print(f"Serialized NTL Images.")
