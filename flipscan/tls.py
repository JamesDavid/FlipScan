"""Self-signed TLS for LAN use.

Phone browsers only expose the microphone (and camera) to secure contexts, so
the voice-clone recorder needs an https:// URL when opened from another device.
A self-signed certificate is enough: accept the browser's warning once and the
context is secure. The cert lives in <root>/.tls/ and is regenerated when it's
missing, expired, or the machine's LAN IPs changed (so the SAN list stays
truthful).
"""

from __future__ import annotations

import datetime
import ipaddress
import socket
from pathlib import Path

VALID_DAYS = 825   # the maximum many clients accept for a leaf cert


def _lan_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:  # the classic default-route trick finds the primary LAN address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips)


def ensure_self_signed(root: Path) -> tuple[Path, Path]:
    """Return (cert_path, key_path), generating/refreshing as needed."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    tdir = root / ".tls"
    cert_path, key_path = tdir / "cert.pem", tdir / "key.pem"
    ips = _lan_ips()

    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            sans = cert.extensions.get_extension_for_class(
                x509.SubjectAlternativeName).value
            have = {str(v) for v in sans.get_values_for_type(x509.IPAddress)}
            fresh = (cert.not_valid_after_utc
                     > datetime.datetime.now(datetime.timezone.utc)
                     + datetime.timedelta(days=7))
            if fresh and set(ips) <= have:
                return cert_path, key_path
        except Exception:
            pass  # unreadable/stale — regenerate below

    tdir.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FlipScan")])
    san = x509.SubjectAlternativeName(
        [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())]
        + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=VALID_DAYS))
            .add_extension(san, critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None),
                           critical=True)
            .sign(key, hashes.SHA256()))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path
