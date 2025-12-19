import os
import types
import argparse
from datetime import date
import pickle
from datetime import datetime
import time
import sys
import importlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt

class BlackMarbleDataset(Dataset):
    def __init__(self, data_dir, case_study, dataset_range=30, size='S', horizon=7, transform=None):
        self.data_dir = data_dir
        self.horizon = horizon
        self.size = size
        self.county_names = sorted(os.listdir(data_dir))
        self.case_study = case_study

        # Sorting each county's images by date
        self.sorted_image_paths = {
          county: find_case_study_dates(
            dataset_range,
            sorted(os.listdir(os.path.join(data_dir, county)),
              key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]), int(x.split('_')[2].split('.')[0]))),
            case_study=case_study  
          )  for county in self.county_names
        }
        
        self.mean = 3.201447427712248
        self.std = 10.389727592468262


        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def denormalize(self, tensor):
      mean = torch.tensor(self.mean).cuda()
      std = torch.tensor(self.std).cuda()

      return tensor * std + mean

    def open_pickle_as_tensor(self, image_path):
      """
      Open pickle file of xarray object with radiance data.
      
      Parameters:
      - image_path (str): path of pickle file

      Returns:
      - data_tensor (torch.tensor): tensor with radiance data
      """

      with open(image_path, 'rb') as file:
        data = pickle.load(file)
      try:
        data_np = data["Gap_Filled_DNB_BRDF-Corrected_NTL"].values
      except KeyError:
        data_np = data.values

      data_np[data_np == 6.5535e+03] = 0
      data_tensor = torch.Tensor(data_np).unsqueeze(0)
      return data_tensor

    def __len__(self):
        return len(self.sorted_image_paths[self.county_names[0]]) - self.horizon * 2

    def __getitem__(self, idx):
        past_image_list = []
        future_image_list = []
        time_embeds_list = []

        # Fetch images for the start_index days period
        for day in range(self.horizon):
            past_days_county_image_list = []  # Hold images for one day from all counties
            future_days_county_image_list = []

            for county in self.county_names:
                county_path = os.path.join(self.data_dir, county)
                past_image_path = os.path.join(
                    county_path, self.sorted_image_paths[county][day + idx])
                future_image_path = os.path.join(
                    county_path, self.sorted_image_paths[county][day + idx +  self.horizon])

                past_image = self.open_pickle_as_tensor(past_image_path)
                future_image = self.open_pickle_as_tensor(future_image_path)

                if self.transform:
                    past_image = self.transform(past_image)
                    future_image = self.transform(future_image)

                past_days_county_image_list.append(past_image)
                future_days_county_image_list.append(future_image)

            time_embed = generate_Date2Vec(past_image_path)
            time_embeds_list.append(time_embed)

            # Stack all county images for one day
            past_image_list.append(torch.stack(past_days_county_image_list))
            future_image_list.append(torch.stack(future_days_county_image_list))

        past_image_tensor = torch.stack(past_image_list)
        future_image_tensor = torch.stack(future_image_list)
        time_embeds = torch.stack(time_embeds_list).view(1, self.horizon, 64).repeat(67, 1, 1) # [67, horizon, 64] 

        return (past_image_tensor, future_image_tensor, time_embeds) 

sys.modules['Model'] = importlib.import_module(__name__)

class Date2VecConvert:
    def __init__(self, model_path="./d2v_model/d2v_98291_17.169918439404636.pth"):
        self.model = torch.load(model_path, map_location='cpu', weights_only=False).eval()
    
    def __call__(self, x):
        with torch.no_grad():
            return self.model.encode(torch.Tensor(x).unsqueeze(0)).squeeze(0).cpu()

class Date2Vec(nn.Module):
    def __init__(self, k=32, act="sin"):
        super(Date2Vec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1
        
        self.fc1 = nn.Linear(6, k1)

        self.fc2 = nn.Linear(6, k2)
        self.d2 = nn.Dropout(0.3)
 
        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(k // 2, 6)
        
        self.fc5 = torch.nn.Linear(6, 6)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], 1)
        return out
d2v = Date2VecConvert(model_path="./d2v_model/d2v_98291_17.169918439404636.pth")

def generate_Date2Vec(filepath):
  """
  Generates time embedding given a file path.
  Paper: https://arxiv.org/abs/1907.05321
  Code: https://github.com/ojus1/Date2Vec

  Parameters:
  - filepath (str): e.g., /dataset/alachua/2022_09_28.pickle 

  Returns:
  - time_embed (torch.tensor)
  """

  year, month, day = os.path.basename(filepath).split('.')[0].split('_')
  
  x = torch.Tensor([[00, 00, 00, int(year), int(month), int(day)]]).float()

  time_embed = d2v(x)
  return time_embed


def find_case_study_dates(dataset_range, image_paths, case_study):
    timestamp_to_image = {pd.Timestamp(image_path.split('.')[0].replace('_', '-')): image_path for image_path in image_paths}
    dates = [pd.Timestamp(image_path.split('.')[0].replace('_', '-')) for image_path in image_paths]
    case_study_indices = [dates.index(date) for date in case_study.values()]
    filtered_dates = set()

    for case_study_index in case_study_indices:
        start_index = case_study_index - dataset_range
        end_index = case_study_index + dataset_range

        case_study_dates = dates[start_index:end_index]

        filtered_dates.update(case_study_dates)
    filtered_image_paths = [timestamp_to_image[date] for date in sorted(filtered_dates)]
    return filtered_image_paths


# Graph WaveNet utilities:

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_adj(filename, adjtype):

    # Load the adjacency matrix from csv file

    if (filename.endswith('.csv')):
        adj_mx = pd.read_csv(filename, index_col=0)
        adj_mx = adj_mx.values
    else:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(filename)

    if adjtype == "doubletransition":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"

    if (filename.endswith('.csv')):
        return None, None, adj
    else:
        return sensor_ids, sensor_id_to_ind, adj

# End of Graph WaveNet utilities.

def ntl_tensor_to_np(ntl, dataset=None, denorm=True):
  if denorm:
    ntl = dataset.denormalize(ntl).cpu()
  
  ntl_np = np.array(ntl)	
  ntl_np = np.transpose(ntl_np, (0, 2, 1))
  ntl_np = np.rot90(ntl_np, k=1, axes=(1, 2))
  ntl_np = ntl_np[0, :, :]
  return ntl_np

def visualize_results_raster(preds, save_dir, save_folder, dataset_dir, dataset):
  """
  Save qualitative results from VST-GNN predictions.
  """
  
  eid_path = os.path.dirname(os.path.dirname(save_dir))

  county_names = sorted(os.listdir(dataset_dir))
  preds_save_dir = os.path.join(eid_path, save_folder)
  os.makedirs(preds_save_dir, exist_ok=True) 
 
  case_study_county_idx = [x for x in range(67)]
 
  for pred_idx in range(preds.shape[0]):
    for pred_horizon in range(preds.shape[2]):

      pred_horizon_folder_path = os.path.join(preds_save_dir, str(pred_horizon + 1))
      os.makedirs(pred_horizon_folder_path, exist_ok=True)

      for county_idx in case_study_county_idx:
        county_horizon_folder_path = os.path.join(pred_horizon_folder_path, county_names[county_idx])
        os.makedirs(county_horizon_folder_path, exist_ok=True)

        pred_input_filename = dataset.sorted_image_paths[county_names[county_idx]][pred_idx + pred_horizon + dataset.horizon].split('.')[0]
        pred_save_path = os.path.join(county_horizon_folder_path, pred_input_filename)

        pred = preds[pred_idx, county_idx, pred_horizon]
        pred_np = ntl_tensor_to_np(pred, dataset)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        c = ax.pcolormesh(pred_np, shading='auto', cmap="cividis")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(pred_save_path, bbox_inches='tight')
        plt.close()


def get_percent_of_normal_ntl(ntl, filename, county_name, month_dir):
  """
  ntl (np.array): radiance data
  filename (str): filename to find correct "normal", i.e., look up the correct month
  county_name (str):
  """
  
  date = datetime.strptime(filename.split('.')[0], "%Y_%m_%d")
  month_composites = load_month_composites(county_name, date, month_dir)
  m_ntl = calculate_average_month_ntl(filename, month_composites)
  pon_ntl = 100 * ( (ntl + 1) / (m_ntl + 1) )
  return pon_ntl
  

def calculate_average_month_ntl(filename, month_composites):
  """
  Calculates the average monthly composite of the last three months from a given date.

  Parameters:
  - filename (str):
  - month_composites (xarray.Dataset): object containing necessary monthly composites

  Returns:
  avg_month_ntl (np.ndarray): represents the last 3 month average ntl
  """

  date = pd.Timestamp(filename.split('.')[0].replace('_', '-'))
  transform = transforms.Resize((128, 128))

  month_1 = (date - pd.DateOffset(months=3)).replace(day=1).strftime("%Y-%m-%d")
  month_2 = (date - pd.DateOffset(months=2)).replace(day=1).strftime("%Y-%m-%d")
  month_3 = (date - pd.DateOffset(months=1)).replace(day=1).strftime("%Y-%m-%d")
  month_list = [month_1, month_2, month_3]

  monthly_ntl = []
  for month in month_list:
    try:
      month_ntl = month_composites["NearNadir_Composite_Snow_Free"].sel(time=month).values
    except Exception:
      month_ntl = month_composites.sel(time=month).values
    month_ntl[month_ntl == 6.5535e+03] = 0
     
    # convert to tensor to use transforms.Resize -> convert back to np
    month_ntl_tensor = transform(torch.Tensor(month_ntl).unsqueeze(0))
    month_ntl = ntl_tensor_to_np(month_ntl_tensor, denorm=False)
    monthly_ntl.append(month_ntl)

  avg_month_ntl = np.mean(monthly_ntl, axis=0)
 
  return avg_month_ntl


def load_month_composites(county_name, date: datetime, base_dir: str):
  """
  Loads all the available monthly composites into memory.

  Parameters:
  - county_name (str): name of county, e.g., 'orange'

  Returns:
  - month_composites (xarray.Dataset): dataset of monthly composites
  """

  county_dir = os.path.join(base_dir, county_name)
  file_path = os.path.join(county_dir, f"{county_name}_{date.year}.pickle")
  with open(file_path, 'rb') as file:
    month_composites = pickle.load(file)

  return month_composites


def visualize_risk_map(ntls, save_dir, save_folder, dataset, month_dir):
  eid_path = os.path.dirname(os.path.dirname(save_dir))

  county_names = sorted(os.listdir(dataset.data_dir))
  save_dir = os.path.join(eid_path, save_folder)
  os.makedirs(save_dir, exist_ok=True)

  case_study_county_idx = [x for x in range(len(county_names))]

  for idx in range(ntls.shape[0]):
    for horizon in range(ntls.shape[2]):

      horizon_folder_path = os.path.join(save_dir, str(horizon + 1))
      os.makedirs(horizon_folder_path, exist_ok=True)

      for county_idx in case_study_county_idx:
        try:
          county_horizon_folder_path = os.path.join(horizon_folder_path, county_names[county_idx])
          os.makedirs(county_horizon_folder_path, exist_ok=True)

          filename = dataset.sorted_image_paths[county_names[county_idx]][idx + horizon + dataset.horizon].split('.')[0]
          save_path = os.path.join(county_horizon_folder_path, filename)
          print(save_path)
          if os.path.exists(f"{save_path}.png"): continue

          ntl = ntls[idx, county_idx, horizon]
          ntl_np = ntl_tensor_to_np(ntl, dataset, denorm=True)
          pon_ntl = get_percent_of_normal_ntl(ntl_np, filename, county_names[county_idx], month_dir)

          # plot using red-yellow-green color map
          fig, ax = plt.subplots(figsize=(10, 10))
          c = ax.pcolormesh(pon_ntl, shading='auto', cmap="RdYlGn", vmin=0, vmax=100)
          plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
          plt.axis("off")
          plt.savefig(save_path, bbox_inches='tight')
          plt.close()
          time.sleep(5)
        except Exception as e:
           print(f"Failed on county {county_names[county_idx]}: {e}")

def _make_case_study(case_study_arg: str) -> dict:
    """
    Returns the dict format expected by find_case_study_dates():
      {"h_<name>": pd.Timestamp(...)}
    """
    predefined = {
        "irma":   ("h_irma",   pd.Timestamp("2017-09-10")),
        "michael":("h_michael",pd.Timestamp("2018-10-10")),
        "ian":    ("h_ian",    pd.Timestamp("2022-09-28")),
        "idalia": ("h_idalia", pd.Timestamp("2023-08-30")),
        "milton": ("h_milton", pd.Timestamp("2024-10-10")),
    }

    key = case_study_arg.lower()
    if key in predefined:
        k, ts = predefined[key]
        return {k: ts}

    # Allow passing an explicit date like "2023-08-30"
    try:
        date.fromisoformat(case_study_arg)
        ts = pd.Timestamp(case_study_arg)
        return {f"h_{case_study_arg}": ts}
    except ValueError as e:
        raise ValueError(
            f"Invalid --case_study '{case_study_arg}'. "
            f"Use one of {list(predefined.keys())} or an ISO date like 2023-08-30."
        ) from e


def _patch_denormalize_device_agnostic(dataset: BlackMarbleDataset):
    """
    utils.BlackMarbleDataset.denormalize() uses .cuda() unconditionally.
    This patch keeps utils.py untouched but makes denormalize work on CPU/CUDA.
    """
    def denormalize_any(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=tensor.device)
        std = torch.tensor(self.std, device=tensor.device)
        return tensor * std + mean

    dataset.denormalize = types.MethodType(denormalize_any, dataset)


def collect_targets_only(
    data_dir: str,
    case_study: dict,
    horizon: int,
    dataset_range: int,
    batch_size: int,
    num_workers: int,
    device: str,
):
    dataset = BlackMarbleDataset(
        data_dir,
        case_study=case_study,
        dataset_range=dataset_range,
        horizon=horizon,
    )

    # Make denorm work regardless of CPU/CUDA without changing utils.py
    _patch_denormalize_device_agnostic(dataset)

    pin = device.startswith("cuda")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    all_targets = []
    with torch.no_grad():
        for batch in loader:
            _, y, _ = batch  # y = future_image_tensor (targets)

            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            # In case a stray dim appears (kept from your original logic)
            if y.dim() == 7:
                y = y.squeeze(4)

            # Dataset gives: [B, horizon, counties, C, H, W]
            # visualize_risk_map expects: [B, counties, horizon, C, H, W]
            y = y.to(device).permute(0, 2, 1, 3, 4, 5)

            all_targets.append(y)

    all_targets = torch.cat(all_targets, dim=0)
    return all_targets, dataset


def main():
    parser = argparse.ArgumentParser(
        description="Visualize risk maps using ONLY Black Marble targets (no model, no checkpoint)."
    )
    parser.add_argument("--case_study", type=str, default="idalia",
                        help="irma/michael/ian/idalia/milton or ISO date like 2023-08-30")
    parser.add_argument("--dir_image", type=str,
                        default="./county_VNP46A2",
                        help="Black Marble county directory (67 subfolders).")
    parser.add_argument("--month_dir_image", type=str,
                        default="./county_VNP46A3",
                        help="Black Marble montly county composite directory (67 subfolders).")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon value used by BlackMarbleDataset.")
    parser.add_argument("--dataset_range", type=int, default=30, help="Days before/after case date to include.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--output_root", type=str, default=".",
                        help="Where to write outputs. (Risk maps go under output_root/<save_folder>/...)")
    parser.add_argument("--save_folder", type=str, default="targets_risk_maps",
                        help="Subfolder name for outputs.")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available — falling back to CPU.")
        device = "cpu"

    case_study = _make_case_study(args.case_study)

    print(f"Dataset dir: {args.dir_image}")
    print(f"Case study:  {case_study}")
    print(f"Horizon:     {args.horizon}")
    print(f"Range:       ±{args.dataset_range} days")
    print(f"Device:      {device}")

    targets, dataset = collect_targets_only(
        data_dir=args.dir_image,
        case_study=case_study,
        horizon=args.horizon,
        dataset_range=args.dataset_range,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # visualize_risk_map() computes eid_path = dirname(dirname(save_dir)).
    # We create a dummy "checkpoint-like" path 2 levels deep so eid_path == output_root.
    dummy_save_dir = os.path.join(args.output_root, "dummy_run", "dummy.ckpt")

    visualize_risk_map(
        targets,
        dummy_save_dir,
        args.save_folder,
        dataset,
        args.month_dir_image
    )

    print(f"\nDone. Outputs written under: {os.path.join(args.output_root, args.save_folder)}")


if __name__ == "__main__":
    main()
