# LLM package for FederatedScope
# Importing workers subpackage so MAP-Elites worker gets registered.
# Wrapped in try-except so worker import failure never breaks the core
# llm package (and thus never breaks Phase 1/Phase 2 experiments).
try:
    from federatedscope.llm import workers  # noqa: F401
except Exception as _e:  # pragma: no cover
    import logging
    logging.getLogger(__name__).warning(
        "MAP-Elites workers import failed (%s); "
        "standard FedAvg experiments are unaffected.", _e)
