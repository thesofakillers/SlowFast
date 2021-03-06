from slowfast.datasets import loader
import slowfast.utils.logging as logging
import slowfast.utils.distributed as du
import slowfast.utils.misc as misc
import slowfast.utils.checkpoint as cu

logger = logging.get_logger(__name__)


def experiment(cfg):
     # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Infer with config:")
    logger.info(cfg)

    # Build the SlowFast model and print its statistics
    model = build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model, cfg, is_train=False)

    # load weights
    if cfg.INFERENCE.WEIGHTS_FILE_PATH != "":
        cu.load_checkpoint(cfg.INFERENCE.WEIGHTS_FILE_PATH, model, cfg.NUM_GPUS > 1, None,
                           inflation=False, convert_from_caffe2=cfg.INFERENCE.WEIGHTS_TYPE == "caffe2")
    else:
        raise FileNotFoundError("Model weights file could not be found")

    perform_inference(model, cfg)
