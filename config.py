# All tunable values live in config.toml. This module loads that file once and
# exposes them as attributes, e.g. CONFIG.KL_COEF, so the rest of the code has
# a single place to look up settings.
import tomllib
from pathlib import Path
from types import SimpleNamespace

CONFIG_PATH = Path(__file__).with_name("config.toml")


def load_config(path=CONFIG_PATH):
    with open(path, "rb") as file:
        sections = tomllib.load(file)

    # The TOML sections ([model], [reward], ...) are just for organization.
    # Flatten them so code can say CONFIG.KL_COEF instead of CONFIG.ppo.kl_coef.
    flat = {}
    for section_values in sections.values():
        for key, value in section_values.items():
            name = key.upper()
            if name in flat:
                raise ValueError(f"Duplicate config key across sections: {key}")
            flat[name] = value
    return SimpleNamespace(**flat)


CONFIG = load_config()
