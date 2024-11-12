from pprint import pprint
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from pprint import pprint
import logging
import wandb
import os
from pathlib import Path
from datetime import datetime
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def eval(conf: DictConfig):
    hydra_cfg = HydraConfig.get()
    experiment_name = hydra_cfg.runtime.choices["experiment"]
    now = datetime.now()
    run_name = f"{experiment_name}/eval"
    output_dir = f"./outputs/{run_name}"
    with open_dict(conf):
        conf.training_arguments.output_dir = output_dir
        conf.training_arguments.run_name = run_name
    resolved_conf = OmegaConf.to_container(conf, resolve=True)
    pprint(resolved_conf)
    os.environ["WANDB_PROJECT"] = conf.wandb.setup.project
    wandb.init(
        group=experiment_name,
        job_type="eval",
        config=resolved_conf,
        resume=conf.wandb.resume,
        settings=wandb.Settings(start_method="fork"),
    )
    logger.info(f"Working directiory: {os.getcwd()}")
    trainer = instantiate(conf.trainer)
    test_dataset = instantiate(conf.test_dataset)
    trainer._load_from_checkpoint(conf.resume_from_checkpoint)
    trainer.predict(test_dataset)
    wandb.finish()


if __name__ == "__main__":
    eval()
