from __future__ import annotations

import pandas as pd
import os
from pathlib import Path
import argparse

def filter_eaglei(
        file: str, 
        state: str = "Florida", 
        output_dir: str = "./florida_data"
    ) -> None:
    file_path = Path(file)
    file_name = file_path.name
    eagle_i = pd.read_csv(file_path)
    florida_counties = eagle_i[(eagle_i["state"] == state)]
    print(florida_counties)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    florida_counties.to_csv(os.path.join(output_dir, file_name))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default="eaglei_outages_2024.csv")
    p.add_argument("--state", type=str, default="Florida")
    p.add_argument("--output-dir", type=str, default="./florida_data")
    args = p.parse_args()

    filter_eaglei(args.file, args.state, args.output_dir)
