from run import get_config_grid, run
from torch.optim import Adam
from pathlib import Path

from src.models import BiTULER

BASEPATH = Path(__file__).parent.parent.joinpath("reports")

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    
    device = "cuda:0"

    path = BASEPATH.joinpath("wall_time.jsonl")
    subsample = None

    configs = []
    for model in [BiTULER]:
        for dataset in ["foursquare_NYC", "foursquare_TKY", "geolife"]:
            n_users = 150 if dataset == "geolife" else 800
            discretization_rows = 800 if dataset == "geolife" else 200
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=[4],
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type=["lookup_weighted_concat_list"],
                discretization_rows=discretization_rows,
                discretization_shape="hex",
                aggregation_mode="grow",
                grow_factor=2,
                lr=2e-4,
                n_hidden=1024,
                n_layers=1,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                subsample=subsample,
                device=device,
                seed=seeds,
                log_path=path,
                use_embedding_offset=False,
            )

            # Baselines
            configs += get_config_grid(
                dataset=dataset,
                model_cls=model,
                n_users=n_users,
                loc_levels=[1],
                time_levels=1,
                optimizer_cls=Adam,
                embedding_type=["lookup_concat"],
                discretization_rows=discretization_rows,
                discretization_shape="hex",
                aggregation_mode="grow",
                grow_factor=2,
                lr=2e-4,
                n_hidden=1024,
                n_layers=1,
                loc_embedding_factor=1,
                time_embedding_factor=1 / 16,
                subsample=subsample,
                device=device,
                seed=seeds,
                log_path=path,
                use_embedding_offset=False,
            )

    # configs = get_missing_configs(
    #     configs, path, relevant_params=["dataset", "embedding_type", "seed"]
    # )
    print(f"Running {len(configs)} configs:")

    for config in configs:
        print(f"{config['dataset']}, {config['embedding_type']}, {config['seed']}")
        run(**config)
