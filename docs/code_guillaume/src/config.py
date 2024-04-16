import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
data_dir = Path.joinpath(project_root, 'data')
results_dir = Path.joinpath(project_root, 'results')

data_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# For pytorch: get device and set dtype
use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')