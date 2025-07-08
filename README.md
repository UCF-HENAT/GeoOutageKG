# GeoOutageKG

GeoOutageKG is a geospatial, multimodal knowledge graph for the domain of power outage detection and analysis in the state of Florida. The corresponding ontology contains separate classes for Outage Records, which is directly recorded from the [Eagle-I](https://doi.org/10.13139/ORNLNCCS/1975202) dataset, NTL Images, from NASA's [Black Marble](https://doi.org/10.1016/j.rse.2018.03.017) dataset, and Outage Map, which is generated using Aparcedo et al.'s [Multimodal Power Outage Prediction](https://doi.org/10.48550/arXiv.2410.00017) model.

Ontologies resued for GeoOutageKG:
- [GEOSatDB](https://doi.org/10.1080/20964471.2024.2331992)
- [DBPedia](https://www.dbpedia.org/)
- [Ontology for Media Resources](https://www.w3.org/ns/ma-ont)

All files including NTL and Outage Map imagery can be found on our OSF repository at https://doi.org/10.17605/OSF.IO/QVD8B.

The ontology for GeoOutageKG, GeoOutageOnto, can be accessed from the web at https://ucf-henat.github.io/GeoOutageOnto/. The repository for that web link can be found [here](https://github.com/UCF-HENAT/GeoOutageOnto).

This project is licensed under the terms of the [MIT License](./LICENSE).

## Installation and Initialization

This section covers how to install all appropirate packages for using the provided data download API (download.py) and knowledge graph generator (GeoOutageKG.py). This section also covers downloading and initializing GeoOutageKG for SPAQRL endpoint use.

### Python:
1. Clone the repository:

    ```
    git clone https://github.com/UCF-HENAT/GeoOutageKG.git
    cd GeoOutageKG
    ```

2. Create virtual enviroment and install required packages
    
    With Anaconda:
    ```
    conda create -n geooutagekg python=3.12 -y
    conda activate geooutagekg
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    With venv:
    ```
    python -m venv geooutagekg
    source geooutagekg/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. (Use if generating your own data, otherwise skip to Step 4) Export NASA Earthdata Token and run download.py (Create Earthdata account and generate token at https://urs.earthdata.nasa.gov/)

    ```
    export EARTH_DATA_TOKEN="EARTH_DATA_TOKEN_HERE"
    python download.py
    ```

4. (Use if using pre-generated data, otherwise skip to Step 3) Download VNP46A2_county_imgs.zip and outage_maps.zip from https://doi.org/10.17605/OSF.IO/QVD8B and unzip contents

    ```
    unzip VNP46A2_county_imgs.zip
    unzip outage_maps.zip
    ```

5. Run GeoOutageKG.py and give root directory as command argument
    ```
    python GeoOutageKG.py ./
    ```

    If necessary, swap './' with root directory containing unzipped VNP46A2_county_imgs/ and outage_maps/ folders.

### SPARQL Endpoint:
1. Download all .ttl files from https://doi.org/10.17605/OSF.IO/QVD8B
2. Import .ttl files into local RDF repository (e.g. GraphDB)

Note: Online SPARQL endpoint for all NTLs, Outage Maps, and Records coming soon!

## Example SPARQL Queries
The following section contains examples of SPARQL queries using GeoOutageKG, showing use cases for the knowledge graph.

### All Data One Month Before and After Hurricane Ian
```SPARQL
PREFIX goo:    <https://ucf-henat.github.io/GeoOutageOnto/#>
PREFIX dbo:    <https://dbpedia.org/ontology/>
PREFIX dbr:    <https://dbpedia.org/page/> 
PREFIX ma-ont: <http://www.w3.org/ns/ma-ont#>
PREFIX schema: <http://schema.org/>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?ntl ?map ?rec
WHERE {
  BIND(dbr:Orange_County%2C_Florida AS ?county)
    
  # raw NTL images
  ?ntl a goo:NTLImage ;
       goo:representsCounty ?county .

  # derived outage maps
  ?map a goo:OutageMap ;
       goo:representsCounty ?county .

  # numerical outage records
  ?rec a goo:OutageRecord ;
       goo:representsCounty ?county ;
       goo:hasNTLImage      ?ntl ;
       goo:hasOutageMap     ?map ;
       goo:recordDateTime   ?date ;
       goo:numberOfOutages  ?num .

  # only “surges” >100 outages in the 1-month windows before and after Hurricane Ian
  FILTER(?num > 100)
  FILTER(
    ?date >= "2022-08-28T00:00:00Z"^^xsd:dateTime &&
    ?date <  "2022-10-28T00:00:00Z"^^xsd:dateTime
  )
}
```
