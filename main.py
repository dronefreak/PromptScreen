import hydra
import uvicorn
from omegaconf import DictConfig

from guards import initialize_guards
from ds_metrics import run_suite

from api import create_app


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    guards = initialize_guards(cfg)

    if cfg.mode == "api":
        print(f"Starting API server at http://{cfg.api.host}:{cfg.api.port}")
        app = create_app(guards)
        uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)

    elif cfg.mode == "evaluate":
        print("Running Data Science evaluation suite...")
        run_suite(cfg, guards)

    else:
        print(f"Error: Unknown mode '{cfg.mode}'. Please use 'api' or 'evaluate'.")


if __name__ == "__main__":
    main()
