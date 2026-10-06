"""Controlled comparison: MiLES at DeeprETA's tuned grid on GeoLife.

Runs the full MiLES embedding (variable per-level dims, learnable level weights)
at DeeprETA's own tuned grid configuration (square grid, loc_levels=3,
grow_factor=4), on GeoLife across all six backbones. Paired with the DeeprETA
runs (reports/TUL_deepreta.jsonl) at the identical grid, this isolates the effect
of MiLES's embedding design (dims + weights) from the grid configuration, and
tests whether MiLES's raw-GeoLife deficit vs DeeprETA is a grid-config artifact.
"""

from pathlib import Path

from torch.optim import Adam

from run import get_config_grid, run_configs, setup_device_config
from src.models import BiTULER, TULHOR, DeepTUL, TULVAE, MainTUL, T3S

BASEPATH = Path(__file__).parent.parent / "reports"

if __name__ == "__main__":
    devices, num_workers, _ = setup_device_config()
    path = BASEPATH / "TUL_miles_dregrid.jsonl"
    lrs = {T3S: 1e-4, TULHOR: 1e-4, DeepTUL: 4e-4}
    seeds = [0, 1, 2, 3, 4]

    configs = []
    for model in [BiTULER, TULHOR, DeepTUL, TULVAE, MainTUL, T3S]:
        configs += get_config_grid(
            dataset="geolife",
            model_cls=model,
            n_users=150,
            loc_levels=3,
            time_levels=1,
            optimizer_cls=Adam,
            embedding_type="lookup_weighted_concat",  # MiLES: dims + learnable weights
            discretization_shape="square",            # DeeprETA's grid
            embedding_weight_factor=2,                 # variable per-level dims
            discretization_rows=1200,
            aggregation_mode="grow",
            grow_factor=4,                             # DeeprETA's tuned grow
            lr=lrs.get(model, 2e-4),
            n_hidden=1024,
            n_layers=2 if model == DeepTUL else 1,
            loc_embedding_factor=1,
            time_embedding_factor=1 / 16,
            subsample=None,
            seed=seeds,
            log_path=path,
        )

    print(f"[eval_miles_dregrid] {len(configs)} configs", flush=True)
    run_configs(configs, devices=devices, num_workers=num_workers, path=path)
    print("[eval_miles_dregrid] done", flush=True)
