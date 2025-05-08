import pandas as pd
from datetime import datetime
from PIL import Image
import os
import json

ntl_dir = "./VNP46A2_county_imgs"
outage_map_dir = "./outage_maps"

def extract_fips_codes():
    eagle_i = pd.read_csv(f"./eagle-i/24237376/florida_data/eaglei_outages_2014.csv")
    outfile = "./eagle-i/24237376/fips_codes.json"
    fips_dict = {}

    for idx, row in eagle_i.iterrows():
        fips_dict[row['county']] = row['fips_code']

    fips_dict = {key: value for key, value in sorted(fips_dict.items())}
    with open(outfile, 'w') as f:
        json.dump(fips_dict, f, indent=4)

def create_outage_records(year: int):
    eagle_i = pd.read_csv(f"./eagle-i/24237376/florida_data/eaglei_outages_{year}.csv")
    outfile = f"./outagerecord_{year}.ttl"

    with open(outfile, 'w') as f:
        f.write(f"""\
@prefix goo: <http://example.org/ontology#> .
@prefix gokg: <http://example.org/resource#> .
@prefix dbr: <https://dbpedia.org/page/> .
@prefix ma-ont: <http://www.w3.org/ns/ma-ont#> .
@prefix eo-ont: <https://www.eoknowledgehub.cn/eo/ontology/> .
@prefix eor: <https://www.eoknowledgehub.cn/eo/resource/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix schema: <http://schema.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

""")

        for idx, entry in eagle_i.iterrows():
            # if idx > 2000:
            #     break
            county_name = entry["county"].replace(" ", "_")
            fips_code = entry["fips_code"]
            try:
                num_outages = entry["customers_out"]
            except KeyError:
                num_outages = entry["sum"]
            time = entry["run_start_time"]
            time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d_%H:%M:%S")
            recordDateTime = datetime.strptime(time, "%Y-%m-%d_%H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")
            date = datetime.strptime(time, "%Y-%m-%d_%H:%M:%S").strftime("%Y-%m-%d")
            date_file = datetime.strptime(date, "%Y-%m-%d").strftime("%Y_%m_%d")
            # print(county_name, fips_code, num_outages, time)
            if os.path.exists(os.path.join(ntl_dir, county_name.lower(), f"{date_file}.png")):
                    hasNTLImage_str = f"\n    goo:hasNTLImage gokg:ntlimage.{fips_code}.{date} ;"
            else:
                hasNTLImage_str = ""

            if os.path.exists(os.path.join(outage_map_dir, county_name.lower(), f"{date_file}.png")):
                    hasOutageMap_str = f"\n    goo:hasOutageMap gokg:outagemap.{fips_code}.{date} ;"
            else:
                hasOutageMap_str = ""

            f.write(f"""\
gokg:outagerecord.{fips_code}.{recordDateTime} a goo:OutageRecord ;
    rdfs:label "{county_name} {recordDateTime}"@en ;
    schema:name "{county_name} {recordDateTime}"@en ;
    rdfs:comment "Eagle-I outage data for {county_name} County at time {recordDateTime}"@en ;
    goo:representsCounty dbr:{county_name}_County%2C_Florida ;{hasNTLImage_str}{hasOutageMap_str}
    goo:recordDateTime "{recordDateTime}"^^xsd:dateTime ;
    goo:numberOfOutages "{num_outages}"^^xsd:integer .

""")
            
def create_ntl_images(data_path: str):
    outfile = f"./ntlimage.ttl"
    with open("./eagle-i/24237376/fips_codes.json", 'r') as f:
        fips_dict = json.load(f)

    with open(outfile, 'w') as f:
        f.write(f"""\
@prefix goo: <http://example.org/ontology#> .
@prefix gokg: <http://example.org/resource#> .
@prefix dbr: <https://dbpedia.org/page/> .
@prefix ma-ont: <http://www.w3.org/ns/ma-ont#> .
@prefix eo-ont: <https://www.eoknowledgehub.cn/eo/ontology/> .
@prefix eor: <https://www.eoknowledgehub.cn/eo/resource/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix schema: <http://schema.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

""")
        
        for dirpath, dirnames, filenames in os.walk(data_path):
            print(dirpath)
            county_name = dirpath.split('\\')[-1].title().replace(" ", "_")
            for file in filenames:
                date = file.split('.')[0]
                date_form = datetime.strptime(date, "%Y_%m_%d").strftime("%Y-%m-%d")
                fips_code = fips_dict[county_name]
                if os.path.exists(os.path.join(outage_map_dir, county_name.lower(), file)):
                    hasOutageMap_str = f"\n    goo:hasOutageMap gokg:outagemap.{fips_code}.{date_form} ;"
                else:
                    hasOutageMap_str = ""
                with Image.open(os.path.join(dirpath, file)) as img:
                    width, height = img.size

                f.write(f"""\
gokg:ntlimage.{fips_code}.{date_form} a goo:NTLImage ;
    rdfs:label "{county_name} {date_form}"@en ;
    schema:name "{county_name} {date_form}"@en ;
    rdfs:comment "Black Marble NTL Image for {county_name} County at date {date_form}"@en ;
    goo:representsCounty dbr:{county_name}_County%2C_Florida ;{hasOutageMap_str}
    ma-ont:frameWidth "{width}"^^xsd:integer ;
    ma-ont:frameHeight "{height}"^^xsd:integer ;
    ma-ont:date "{date_form}"^^xsd:date ;
    ma-ont:locator <http://127.0.0.1:5000/ntlimage/{county_name.lower()}/{date}.png> , <http://127.0.0.1:5000/ntlimage/{county_name.lower()}/{date}.pkl> .

""")
                
def create_outage_maps(data_path: str):
    outfile = f"./outagemap.ttl"
    with open("./eagle-i/24237376/fips_codes.json", 'r') as f:
        fips_dict = json.load(f)

    with open(outfile, 'w') as f:
        f.write(f"""\
@prefix goo: <http://example.org/ontology#> .
@prefix gokg: <http://example.org/resource#> .
@prefix dbr: <https://dbpedia.org/page/> .
@prefix ma-ont: <http://www.w3.org/ns/ma-ont#> .
@prefix eo-ont: <https://www.eoknowledgehub.cn/eo/ontology/> .
@prefix eor: <https://www.eoknowledgehub.cn/eo/resource/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix schema: <http://schema.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

""")
        
        for dirpath, dirnames, filenames in os.walk(data_path):
            print(dirpath)
            county_name = dirpath.split('\\')[-1].title().replace(" ", "_")
            for file in filenames:
                date = file.split('.')[0]
                date_form = datetime.strptime(date, "%Y_%m_%d").strftime("%Y-%m-%d")
                fips_code = fips_dict[county_name]
                with Image.open(os.path.join(dirpath, file)) as img:
                    width, height = img.size

                f.write(f"""\
gokg:outagemap.{fips_code}.{date_form} a goo:OutageMap ;
    rdfs:label "{county_name} {date_form}"@en ;
    schema:name "{county_name} {date_form}"@en ;
    rdfs:comment "Outage Map for {county_name} County at date {date_form}"@en ;
    goo:representsCounty dbr:{county_name}_County%2C_Florida ;
    ma-ont:frameWidth "{width}"^^xsd:integer ;
    ma-ont:frameHeight "{height}"^^xsd:integer ;
    ma-ont:date "{date_form}"^^xsd:date ;
    ma-ont:locator <http://127.0.0.1:5000/outagemap/{county_name.lower()}/{date}.png> .

""")
                
            
if __name__ == "__main__":
    for year in range(2014, 2025):
        create_outage_records(year)
    create_outage_maps(outage_map_dir)
    create_ntl_images(ntl_dir)