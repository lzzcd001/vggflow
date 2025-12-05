from config.default_config import get_default_configs

def get_config():
    config = get_default_configs()
    config.experiment.prompt_fn = "pap" # for Pick-a-Pic
    config.experiment.reward_fn = "pickscore"
    config.experiment.prompt_fn_kwargs = {}

    return config