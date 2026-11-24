"""
MAP-Elites workers for Federated RLHF.

Registers the 'map_elites' federated method which triggers
MAPElitesServer + MAPElitesClient instead of the default Server + Client.

Triggered by cfg.federate.method = 'map_elites' in YAML.
"""
from federatedscope.register import register_worker
from federatedscope.llm.workers.map_elites_server import MAPElitesServer
from federatedscope.llm.workers.map_elites_client import MAPElitesClient


def call_map_elites_worker(method):
    """Register: when method == 'map_elites', use MAP-Elites workers."""
    if method == 'map_elites':
        return {'server': MAPElitesServer, 'client': MAPElitesClient}
    return None


register_worker('map_elites', call_map_elites_worker)
