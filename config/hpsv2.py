from config.default_config import get_default_configs

def get_config():
    config = get_default_configs()
    config.experiment.prompt_fn = "hpd_photo_painting" # for HPSv2
    config.experiment.reward_fn = "hpscore"
    config.experiment.prompt_fn_kwargs = {}

    return config