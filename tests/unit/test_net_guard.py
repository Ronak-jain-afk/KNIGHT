# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import httpx
import pytest

from app.security.net_guard import (
    BlockedAddressError,
    InvalidURLError,
    NetGuard,
    PinnedNetworkBackend,
    pinned_transport,
)


@pytest.mark.parametrize(
    "blocked_host",
    [
        "127.0.0.1",
        "localhost",
        "::1",
        "0.0.0.0",
        "10.0.0.1",
        "172.16.0.1",
        "172.31.255.254",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "fe80::1",
        "fc00::1",
    ],
)
def test_net_guard_blocks_prohibited_ips(blocked_host):
    url = f"http://{blocked_host}/status"
    # When testing ipv6 in url brackets
    if ":" in blocked_host and not blocked_host.startswith("["):
        url = f"http://[{blocked_host}]/status"

    with pytest.raises(BlockedAddressError):
        NetGuard.validate_url(url)


@pytest.mark.parametrize(
    "bad_url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file.txt",
        "gopher://example.com/",
        "javascript:alert(1)",
        "mailto:user@example.com",
    ],
)
def test_net_guard_blocks_unsupported_schemes(bad_url):
    with pytest.raises(InvalidURLError, match="Unsupported scheme"):
        NetGuard.validate_url(bad_url)


def test_net_guard_blocks_userinfo():
    with pytest.raises(InvalidURLError, match="Userinfo credentials"):
        NetGuard.validate_url("http://user:pass@example.com/api")


def test_net_guard_blocks_non_allowlisted_ports():
    with pytest.raises(InvalidURLError, match="Port 8080 is not allowlisted"):
        NetGuard.validate_url("http://example.com:8080")

    with pytest.raises(InvalidURLError, match="Port 22 is not allowlisted"):
        NetGuard.validate_url("http://example.com:22")


def test_net_guard_allows_public_https_and_8443():
    # Mock DNS resolution for example.com to public IP (93.184.216.34)
    fake_addr_info = [(2, 1, 6, "", ("93.184.216.34", 443))]
    with patch("socket.getaddrinfo", return_value=fake_addr_info):
        host, port = NetGuard.validate_url("https://example.com")
        assert host == "example.com"
        assert port == 443

    fake_addr_info_8443 = [(2, 1, 6, "", ("93.184.216.34", 8443))]
    with patch("socket.getaddrinfo", return_value=fake_addr_info_8443):
        host, port = NetGuard.validate_url("https://example.com:8443/article")
        assert host == "example.com"
        assert port == 8443


def test_resolve_url_returns_validated_ip():
    """H-01: the address used for the connection is the one that was checked."""
    fake_addr_info = [(2, 1, 6, "", ("93.184.216.34", 443))]
    with patch("socket.getaddrinfo", return_value=fake_addr_info):
        hostname, port, ip = NetGuard.resolve_url("https://example.com/article")
    assert (hostname, port, ip) == ("example.com", 443, "93.184.216.34")


class _RecordingBackend:
    def __init__(self) -> None:
        self.connected: list[tuple[str, int]] = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connected.append((host, port))
        return object()

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return object()

    async def sleep(self, seconds):
        return None


@pytest.mark.asyncio
async def test_pinned_backend_ignores_dns_flip():
    """H-01: a resolver that flips to loopback after validation cannot redirect us."""
    answers = [
        [(2, 1, 6, "", ("93.184.216.34", 443))],
        [(2, 1, 6, "", ("127.0.0.1", 443))],
    ]

    with patch("socket.getaddrinfo", side_effect=answers):
        hostname, _port, pinned_ip = NetGuard.resolve_url("https://example.com/")
        # Simulates httpx resolving the hostname again (the rebinding attempt).
        with patch("socket.getaddrinfo", return_value=answers[1]):
            backend = _RecordingBackend()
            pinned = PinnedNetworkBackend({hostname: pinned_ip}, backend=backend)
            await pinned.connect_tcp(hostname, 443)

    assert pinned_ip == "93.184.216.34"
    assert backend.connected == [("93.184.216.34", 443)]


def test_pinned_transport_does_not_use_env_proxy(monkeypatch):
    """H-02: an HTTPS_PROXY in the environment cannot take over the guarded client."""
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    transport = pinned_transport("example.com", "93.184.216.34")
    client = httpx.AsyncClient(transport=transport, trust_env=False)

    chosen = client._transport_for_url(httpx.URL("https://example.com/article"))
    assert chosen is transport
    assert isinstance(transport._pool._network_backend, PinnedNetworkBackend)

