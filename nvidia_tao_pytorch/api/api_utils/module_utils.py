import importlib
import pkg_resources

entry_points = [ep for ep in pkg_resources.iter_entry_points('console_scripts') if ep.module_name.startswith('nvidia_tao_pytorch')]

def get_entry_points():
    eps = [ep.name for ep in entry_points]
    return eps

def get_neural_network_actions(neural_network_name):
    for ep in entry_points:
        if ep.name == neural_network_name:
            module = importlib.import_module(ep.module_name)
            actions = module.get_subtask_list()
            return module, actions
    return None, None

def get_entry_point_module_mapping():
    return {ep.name:ep.module_name for ep in entry_points}
