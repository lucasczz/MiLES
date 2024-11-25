from src.data.utils import get_dataloader
from src.models.deeptul import DeepTUL
from src.models.maintul import MainTUL
from src.models.tulvae import TULVAE
from tools.run import grid_search, run
from torch.optim import Adam


if __name__ == "__main__":
    batch_size = 1
    dataset= 'foursquare_TKY'
    n_users = 400
    device=  "cuda:0"
    subsample = None
    log_path= 'debug_deeptul'
    # Get the dataloader and other dataset-related information
    dataloader, n_locs, n_times = get_dataloader(
        dataset, n_users, batch_size, device, subsample
    )
  
    run(
        model_cls=DeepTUL,
        dataset_name=dataset,
        dataloader=dataloader,
        n_locs=n_locs,
        n_times=n_times,
        n_users=n_users,
        device=device,
        log_path=log_path,
        verbose=True,
    )
