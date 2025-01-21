import importlib

def load_config_from_file(file_path):
    spec = importlib.util.spec_from_file_location("config_module", file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

