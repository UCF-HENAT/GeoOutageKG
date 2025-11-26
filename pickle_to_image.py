import os
import pickle
import matplotlib.pyplot as plt
import argparse

def plot_vnp46a2(directory: str, overwrite: bool = False):
    # Specify the path to your pickle file.
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            try:
                if (file.endswith('.pkl') or file.endswith('.pickle')):
                    pickle_file = os.path.join(dirpath, file)
                    print(pickle_file)

                    # Check if the file exists.
                    if not os.path.exists(pickle_file):
                        raise FileNotFoundError(f"The file '{pickle_file}' does not exist.")
                    
                    image_folder = f"{directory}_imgs"
                    output_dir = os.path.join(image_folder, dirpath.split('/')[-1])
                    os.makedirs(output_dir, exist_ok=True)
                    image_path = os.path.join(output_dir, file.replace('pkl', 'png').replace('pickle', 'png'))
                    if not overwrite and os.path.exists(image_path):
                        print("Already exists, overwrite flag not set...")
                        continue

                    # Load the pickle file (which should contain an xarray.Dataset).
                    with open(pickle_file, 'rb') as f:
                        ds = pickle.load(f)

                    # if variable_to_plot in ds:
                    # Create a figure for the plot.
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Use xarray's built-in plotting functionality.
                    # The 'robust=True' option helps in cases with outliers by using percentiles.
                    ds.plot(
                        ax=ax,
                        cmap='gray',
                        robust=True,
                        add_colorbar=False,
                        add_labels=False  # This option can sometimes remove default labels depending on the xarray version.
                    )
                    
                    # Customize the plot.
                    ax.axis('off')

                    # Set the aspect ratio to equal so that one unit in x is the same as one unit in y.
                    ax.set_aspect('equal', adjustable='box')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    
                    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

            except Exception as e:
                print(e)

def plot_vnp46a3(directory: str, overwrite: bool = False):
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            try:
                if file.endswith('.pkl') or file.endswith('.pickle'):
                    pickle_file = os.path.join(dirpath, file)
                    print(pickle_file)

                    # Check if the file exists.
                    if not os.path.exists(pickle_file):
                        raise FileNotFoundError(f"The file '{pickle_file}' does not exist.")
                    
                    image_folder = f"{directory}_imgs"
                    output_dir = os.path.join(image_folder, dirpath.split('/')[-1])
                    os.makedirs(output_dir, exist_ok=True)
                    image_path = os.path.join(output_dir, file.replace('pkl', 'png').replace('pickle', 'png'))
                    if not overwrite and os.path.exists(image_path):
                        print("Already exists, overwrite flag not set...")
                        continue

                    # Load the pickle file (which should contain an xarray.Dataset).
                    with open(pickle_file, 'rb') as f:
                        ds = pickle.load(f)

                    # Create a figure for the plot.
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Use xarray's built-in plotting functionality.
                    # The 'robust=True' option helps in cases with outliers by using percentiles.
                    ds[0].plot(
                        ax=ax,
                        cmap='gray',
                        robust=True,
                        add_colorbar=False,
                        add_labels=False  # This option can sometimes remove default labels depending on the xarray version.
                    )
                    
                    # Customize the plot.
                    ax.axis('off')

                    # Set the aspect ratio to equal so that one unit in x is the same as one unit in y.
                    ax.set_aspect('equal', adjustable='box')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    
                    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

            except Exception as e:
                print(e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="./county_VNP46A2")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if "VNP46A2" in args.dir:
        plot_vnp46a2(args.dir, args.overwrite)
    elif "VNP46A3" in args.dir:
        plot_vnp46a3(args.dir, args.overwrite)
    else:
        print("Invalid product given.")