import hydra
import uvicorn
from omegaconf import DictConfig

from guards import initialize_guards, initialize_all_guards
from ds_metrics import run_suite
from pipeline import evaluate

from api import create_app


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    if cfg.mode == "api":
        guards = initialize_all_guards(cfg)
        print(f"Starting API server at http://{cfg.api.host}:{cfg.api.port}")
        app = create_app(guards)
        uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)

    elif cfg.mode == "stats":
        print("Running Data Science evaluation suite...")
        guards = initialize_guards(cfg)
        run_suite(cfg, guards)

    elif cfg.mode == "pipeline":
        print("Running pipeline with configured subset of defences")
        guards = initialize_guards(cfg)
        evaluate(cfg, guards)

    else:
        print(f"Error: Unknown mode '{cfg.mode}'. Please use 'api', 'stats' or 'pipeline'.")


if __name__ == "__main__":
    main()
