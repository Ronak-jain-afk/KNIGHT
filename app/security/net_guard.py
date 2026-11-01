# SPDX-License-Identifier: Apache-2.0
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from httpcore import AnyIOBackend, AsyncNetworkBackend

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443, 8443}

# CGNAT network: 100.64.0.0/10
CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


class NetGuardError(Exception):
    """Base exception for NetGuard violations."""


class BlockedAddressError(NetGuardError):
    """Raised when an address or host resolves to a blocked IP range."""


class InvalidURLError(NetGuardError):
    """Raised when a URL scheme, credentials, or port is invalid."""


class NetGuard:
    """SSRF protection guard validating schemes, credentials, ports, and resolved IPs."""

    @classmethod
    def is_ip_blocked(cls, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        if ip.is_loopback:
            return True
        if ip.is_private:
            return True
        if ip.is_link_local:
            return True
        if ip.is_multicast:
            return True
        if ip.is_reserved:
            return True
        if ip.is_unspecified:
            return True
        return bool(isinstance(ip, ipaddress.IPv4Address) and ip in CGNAT_NETWORK)

    @classmethod
    def resolve(cls, url: str) -> tuple[str, int, list[str]]:
        """Validate ``url`` and return its host, port, and validated IP literals.

        The addresses are resolved exactly once so callers can pin the connection
        to the validated IP (closing the DNS-rebinding TOCTOU window).
        """
        parsed = urlparse(url.strip())

        # Validate scheme
        if not parsed.scheme or parsed.scheme.lower() not in ALLOWED_SCHEMES:
            raise InvalidURLError(
                f"Unsupported scheme '{parsed.scheme}'. Only http and https are allowed."
            )

        # Disallow credentials in URL
        if parsed.username or parsed.password:
            raise InvalidURLError("Userinfo credentials in URL are prohibited.")

        hostname = parsed.hostname
        if not hostname:
            raise InvalidURLError("URL must contain a valid hostname.")

        # Determine port
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme.lower() == "https" else 80

        if port not in ALLOWED_PORTS:
            raise InvalidURLError(
                f"Port {port} is not allowlisted (allowed: {sorted(ALLOWED_PORTS)})."
            )

        # Resolve host and validate all returned IP addresses
        try:
            addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise BlockedAddressError(f"Could not resolve host '{hostname}': {exc}") from exc

        if not addr_info:
            raise BlockedAddressError(f"No IP addresses resolved for '{hostname}'.")

        addresses: list[str] = []
        for entry in addr_info:
            ip_str = entry[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError as exc:
                raise BlockedAddressError(f"Invalid resolved IP '{ip_str}': {exc}") from exc

            if cls.is_ip_blocked(ip):
                raise BlockedAddressError(
                    f"Address '{hostname}' resolved to blocked IP '{ip_str}'."
                )
            addresses.append(ip_str)

        return hostname, port, addresses

    @classmethod
    def validate_url(cls, url: str) -> tuple[str, int]:
        """Validate URL scheme, userinfo, port, and resolved IP addresses.

        Returns (hostname, port).
        Raises InvalidURLError or BlockedAddressError.
        """
        hostname, port, _ = cls.resolve(url)
        return hostname, port

    @classmethod
    def resolve_url(cls, url: str) -> tuple[str, int, str]:
        """Validate ``url`` and return ``(hostname, port, pinned_ip)``.

        Callers must connect to ``pinned_ip`` (with the hostname preserved for
        Host/SNI) instead of letting httpx resolve the hostname a second time.
        """
        hostname, port, addresses = cls.resolve(url)
        return hostname, port, addresses[0]


class PinnedNetworkBackend(AsyncNetworkBackend):
    """Network backend that connects to pre-validated IPs for allowed hosts.

    httpcore resolves the hostname once by itself, which reopens the rebinding
    TOCTOU that ``NetGuard`` closed. This backend redirects each mapped host to
    the IP that was validated up front while httpcore keeps the original host for
    the ``Host`` header and TLS SNI.
    """

    def __init__(
        self,
        pinned: dict[str, str],
        backend: AsyncNetworkBackend | None = None,
    ) -> None:
        self._pinned = {host.lower(): ip for host, ip in pinned.items()}
        self._backend = backend or AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: list[tuple[int, int, int | bytes]] | None = None,
    ):
        target = self._pinned.get(host.lower(), host)
        return await self._backend.connect_tcp(
            target,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: list[tuple[int, int, int | bytes]] | None = None,
    ):
        return await self._backend.connect_unix_socket(
            path, timeout=timeout, socket_options=socket_options
        )

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


def pinned_transport(hostname: str, ip: str, **kwargs) -> httpx.AsyncHTTPTransport:
    """Build an httpx transport that only ever connects to ``ip`` for ``hostname``.

    httpx does not expose httpcore's ``network_backend`` hook, so the pool is
    rewired after construction. Proxy auto-detection is disabled so an
    environment proxy cannot bypass the pinned route (H-02).
    """
    kwargs.setdefault("trust_env", False)
    transport = httpx.AsyncHTTPTransport(**kwargs)
    pool = getattr(transport, "_pool", None)
    if pool is None:  # pragma: no cover - defensive against httpx refactors
        raise RuntimeError("httpx transport does not expose a connection pool.")
    pool._network_backend = PinnedNetworkBackend({hostname: ip})
    return transport
