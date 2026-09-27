"""Execution-layer configuration and connectivity errors."""


class ConfigurationError(Exception):
    """Raised when broker credentials or connectivity checks fail at startup.

    Callers (runner, API) should treat this as a hard misconfiguration: do not
    substitute a stub broker or continue a trading cycle.
    """


class OrderNotFoundError(LookupError):
    """Broker no longer has this order id. Callers must not invent a fill."""


class DuplicateClientOrderIdError(RuntimeError):
    """Raised when a ``client_order_id`` was already submitted recently.

    Defense-in-depth against multiple schedulers (or rapid retries) submitting
    the same logical order. The broker's own dedupe is the first line; this is
    the local backstop when the broker accepts duplicates anyway.
    """
