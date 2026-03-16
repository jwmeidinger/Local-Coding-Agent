from __future__ import annotations

import logging


def apply_source_ip_binding(source_ip: str) -> None:
    """Monkey-patch connection-creation functions to bind ALL outgoing TCP
    connections to the given local IP address.  This forces traffic through
    the correct network interface (e.g. Wi-Fi) and bypasses VPN default routes.

    Call this once at startup, before any HTTP clients are created.

    We patch THREE levels to be thorough:
      1. socket.create_connection        — used by stdlib http.client / urllib
      2. urllib3.util.connection.create_connection — used by requests / LangChain
      3. urllib3.util.connection.HAS_IPV6 — force IPv4 so we don't try to
         bind an IPv4 source_address to an IPv6 socket
    """
    logger = logging.getLogger("coding-agent")

    # --- 1. Patch stdlib socket.create_connection ---
    import socket as _socket
    _orig_socket_create = _socket.create_connection

    def _bound_socket_create(address, timeout=_socket._GLOBAL_DEFAULT_TIMEOUT,
                              source_address=None, **kwargs):
        if source_address is None:
            source_address = (source_ip, 0)
        return _orig_socket_create(address, timeout, source_address, **kwargs)

    _socket.create_connection = _bound_socket_create
    logger.info("Patched socket.create_connection → source_address=%s", source_ip)

    # --- 2. Patch urllib3's create_connection (used by requests & LangChain) ---
    try:
        import urllib3.util.connection as _u3conn

        _orig_u3_create = _u3conn.create_connection

        def _bound_u3_create(address, timeout=_socket._GLOBAL_DEFAULT_TIMEOUT,
                              source_address=None, socket_options=None):
            if source_address is None:
                source_address = (source_ip, 0)
            return _orig_u3_create(address, timeout, source_address, socket_options)

        _u3conn.create_connection = _bound_u3_create

        # Force IPv4 so urllib3 doesn't try AF_INET6 with our IPv4 source address
        _u3conn.HAS_IPV6 = False

        logger.info("Patched urllib3.util.connection.create_connection → source_address=%s", source_ip)
    except (ImportError, AttributeError) as e:
        logger.warning("Could not patch urllib3 (will fall back to socket-level patch): %s", e)

    logger.info("Global source-IP binding active: all outgoing connections will use %s", source_ip)
