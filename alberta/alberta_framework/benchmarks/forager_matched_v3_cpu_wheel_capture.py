"""Explicitly networked, nonauthorizing CPU-wheel capture for matched-v3.

This boundary consumes the canonical *untrusted* capture manifest accepted by
``forager_matched_v3_cpu_wheelhouse``.  It requests each exact HTTPS origin,
rejects redirects, and checks only that the received bytes match the size and
SHA-256 commitments in that untrusted manifest.  Those checks do not establish
package authenticity, content safety, TLS/index/resolver trust, network
isolation, runtime qualification, scientific evidence, or execution authority.

Publication has no default path.  A caller must explicitly authorize network
capture and supply an existing absolute root.  A private directory is populated
through retained directory descriptors and atomically published, new-only, as
``sha256/<manifest-full-sha256>/{manifest.v1.json,wheels/}``.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import hmac
import http.client
import math
import os
import re
import secrets
import ssl
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn, Protocol, cast
from urllib.parse import urlsplit

from alberta_framework.benchmarks import forager_matched_v3_cpu_wheelhouse as _wheelhouse

CPU_WHEEL_CAPTURE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_wheel_capture_contract.v1"
)
CPU_WHEEL_CAPTURE_STATUS: Final = "implemented_networked_untrusted_non_authorizing"
CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION: Final = (
    _wheelhouse.CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION
)

_NAMESPACE: Final = "sha256"
_MANIFEST_FILENAME: Final = "manifest.v1.json"
_WHEELS_DIRECTORY: Final = "wheels"
_READ_CHUNK_BYTES: Final = 1024 * 1024
_DEFAULT_REQUEST_TIMEOUT_SECONDS: Final = 30.0
_MAX_REQUEST_TIMEOUT_SECONDS: Final = 300.0
_PYPI_WHEEL_ORIGIN_RE: Final = re.compile(
    r"https://files\.pythonhosted\.org/packages/"
    r"[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{60}/"
    r"(?P<filename>[A-Za-z0-9][A-Za-z0-9_.+\-]{0,250}\.whl)\Z"
)


class ForagerMatchedV3CpuWheelCaptureError(_wheelhouse.ForagerMatchedV3CpuWheelhouseError):
    """Capture failed; a reported post-commit close error retains the exact publication."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3CpuWheelCaptureError(message)


class HTTPSCaptureResponse(Protocol):
    """Minimal streaming response surfaced by an injected HTTPS transport."""

    status_code: int
    final_url: str

    def read(self, size: int) -> bytes:
        """Read at most ``size`` response bytes, returning ``b""`` at EOF."""

    def close(self) -> None:
        """Release the response and its underlying connection."""


class HTTPSCaptureTransport(Protocol):
    """Transport contract used for exact-origin HTTPS GET requests."""

    def open(self, url: str, *, timeout_seconds: float) -> HTTPSCaptureResponse:
        """Open one response without following redirects."""


class _StdlibResponse:
    def __init__(
        self,
        connection: http.client.HTTPSConnection,
        response: http.client.HTTPResponse,
        requested_url: str,
    ) -> None:
        self._connection = connection
        self._response = response
        self.status_code = response.status
        self.final_url = requested_url

    def read(self, size: int) -> bytes:
        return self._response.read(size)

    def close(self) -> None:
        failure: BaseException | None = None
        try:
            self._response.close()
        except BaseException as exc:
            failure = exc
        try:
            self._connection.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"connection close also failed: {exc!r}")
        if failure is not None:
            raise failure


class StdlibHTTPSCaptureTransport:
    """Small standard-library HTTPS transport that follows zero redirects.

    The default SSL context is an operational mechanism, not an attestation of
    the peer, resolver, package index, or captured content.  Redirect responses
    are returned to the boundary and rejected there.
    """

    def __init__(self) -> None:
        self._context = ssl.create_default_context()

    def open(self, url: str, *, timeout_seconds: float) -> HTTPSCaptureResponse:
        parsed = urlsplit(url)
        try:
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise ForagerMatchedV3CpuWheelCaptureError(
                "HTTPS origin contains an invalid authority"
            ) from exc
        if (
            parsed.scheme != "https"
            or hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            _fail("transport received a noncanonical HTTPS origin")
        target = parsed.path or "/"
        connection = http.client.HTTPSConnection(
            hostname,
            port=port,
            timeout=timeout_seconds,
            context=self._context,
        )
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Accept": "application/octet-stream",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                    "User-Agent": "alberta-matched-v3-untrusted-wheel-capture/1",
                },
            )
            response = connection.getresponse()
        except BaseException:
            connection.close()
            raise
        return _StdlibResponse(connection, response, url)


@dataclass(frozen=True, slots=True)
class PublishedMatchedV3CpuWheelCapture:
    """Paths and byte counts for one new-only nonauthorizing publication."""

    directory: Path
    manifest: Path
    wheels: Path
    manifest_sha256: str
    wheel_count: int
    total_size_bytes: int


def _claims() -> dict[str, bool]:
    descriptor = _wheelhouse.cpu_wheelhouse_contract_descriptor()
    inherited = cast(dict[str, Any], descriptor["claims"])
    claims = {key: False for key in sorted(inherited)}
    claims.update(
        {
            "content_verified": False,
            "index_trusted": False,
            "resolver_trusted": False,
            "tls_peer_trusted": False,
            "transport_implementation_attested": False,
        }
    )
    return dict(sorted(claims.items()))


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": CPU_WHEEL_CAPTURE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
        "status": CPU_WHEEL_CAPTURE_STATUS,
        "classification": "network_capture_manifest_matched_nonauthorizing",
        "schemas": {"capture_manifest": CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION},
        "network": {
            "implemented_here": True,
            "explicit_authorization_required": True,
            "scheme": "https",
            "method": "GET",
            "redirects_followed": 0,
            "credentials_allowed": False,
            "query_allowed": False,
            "fragment_allowed": False,
            "exact_authority": "files.pythonhosted.org",
            "explicit_port_allowed": False,
            "origin_path": "packages/<2-lower-hex>/<2-lower-hex>/<60-lower-hex>/<filename>",
            "origin_basename_equals_manifest_filename": True,
            "injected_transport_supported": True,
            "injected_transport_behavior_attested": False,
        },
        "matching": {
            "source_of_size_and_sha256": "canonical_untrusted_capture_manifest",
            "exact_size_required": True,
            "exact_sha256_required": True,
            "package_content_authenticated": False,
        },
        "publication": {
            "caller_supplied_absolute_root": True,
            "default_root": False,
            "content_address": "sha256/<capture-manifest-full-sha256>",
            "members": ["manifest.v1.json", "wheels/"],
            "wheel_layout": "flat_original_filenames",
            "new_only": True,
            "overwrite_allowed": False,
            "post_commit_descriptor_close_failure": (
                "reported_with_committed_content_address_preserved"
            ),
        },
        "claims": _claims(),
        "limitations": [
            "HTTPS use does not establish TLS peer or origin trust.",
            "Package-index identity and resolver provenance are not trusted or attested.",
            "No network isolation, route isolation, or DNS isolation is claimed.",
            (
                "Origin syntax is restricted to exact files.pythonhosted.org production wheel "
                "paths, but DNS, routing, CA, CDN, and peer behavior remain external trust."
            ),
            (
                "Matching bytes to an untrusted manifest commitment is transport capture "
                "matching, not package authenticity, safety, or content verification."
            ),
            (
                "The transport implementation is caller-observable but not authenticated, "
                "isolated, or attested by this boundary."
            ),
            (
                "The caller must exclude concurrent mutation of the publication root by another "
                "writer; retained identity checks detect substitutions after entries are opened."
            ),
            (
                "A failure before a newly made directory is opened may leave that private empty "
                "entry behind; cleanup will not delete an unproven pathname after the fact."
            ),
            (
                "A directory-descriptor close error after atomic commit is reported as an error "
                "while the validated new-only content address remains published."
            ),
            (
                "Capture grants no artifact acceptance, runtime qualification, scientific "
                "evidence, publication authority, installation authority, or execution authority."
            ),
        ],
    }


def cpu_wheel_capture_contract_descriptor() -> dict[str, Any]:
    """Return a detached descriptor whose authority claims are all false."""

    return copy.deepcopy(_descriptor())


def _inode(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _open_absolute_directory(path: Path, *, label: str) -> tuple[int, tuple[int, int]]:
    if type(path) is not type(Path()) or not path.is_absolute():
        _fail(f"{label} must be one exact absolute pathlib.Path")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ForagerMatchedV3CpuWheelCaptureError(f"cannot stat {label}") from exc
    if not stat.S_ISDIR(before.st_mode):
        _fail(f"{label} must be an existing non-symlink directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatchedV3CpuWheelCaptureError(
            f"{label} must be an existing non-symlink directory"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        located = path.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _inode(before) != _inode(opened)
            or _inode(opened) != _inode(located)
        ):
            _fail(f"{label} identity changed while opened")
        return descriptor, _inode(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[int, tuple[int, int]]:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ForagerMatchedV3CpuWheelCaptureError(f"cannot stat {label}") from exc
    if not stat.S_ISDIR(before.st_mode):
        _fail(f"{label} is not a non-symlink directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ForagerMatchedV3CpuWheelCaptureError(
            f"{label} is not an openable non-symlink directory"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        located = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _inode(before) != _inode(opened)
            or _inode(opened) != _inode(located)
        ):
            _fail(f"{label} identity changed while opened")
        return descriptor, _inode(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _create_directory_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[int, tuple[int, int]]:
    os.mkdir(name, 0o700, dir_fd=parent_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    created_identity: tuple[int, int] | None = None
    failure: BaseException | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        created_identity = _inode(opened)
        located = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(located.st_mode)
            or _inode(located) != created_identity
        ):
            _fail(f"{label} identity changed after creation")
        os.fchmod(descriptor, 0o700)
        return descriptor, created_identity
    except BaseException as exc:
        failure = exc
    if descriptor >= 0 and created_identity is None:
        try:
            recovered = os.fstat(descriptor)
            if not stat.S_ISDIR(recovered.st_mode):
                _fail(f"{label} retained creation descriptor is not a directory")
            created_identity = _inode(recovered)
        except BaseException as exc:
            failure.add_note(f"{label} identity recovery also failed: {exc!r}")
    if created_identity is not None:
        try:
            _remove_owned_empty_directory_at(
                parent_fd,
                name,
                expected_identity=created_identity,
                label=label,
            )
        except BaseException as exc:
            failure.add_note(f"{label} partial-create cleanup also failed: {exc!r}")
    else:
        failure.add_note(
            f"{label} partial-create cleanup was skipped because no retained identity exists"
        )
    if descriptor >= 0:
        try:
            os.close(descriptor)
        except OSError as exc:
            failure.add_note(f"{label} descriptor close also failed: {exc!r}")
    raise failure


def _remove_owned_empty_directory_at(
    parent_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    """Remove only one still-named empty directory created by this call."""

    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    failure: BaseException | None = None
    try:
        opened = os.fstat(descriptor)
        located = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _inode(opened) != expected_identity
            or _inode(located) != expected_identity
            or os.listdir(descriptor)
        ):
            _fail(f"{label} partial-create cleanup path changed or is not empty")
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except BaseException as exc:
        failure = exc
    _close_preserving_failure(
        descriptor,
        label=f"{label} partial-create cleanup descriptor",
        failure=failure,
    )


def _write_all(descriptor: int, raw: bytes, *, label: str) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            _fail(f"{label} write made no progress")
        view = view[written:]


def _close_preserving_failure(
    descriptor: int,
    *,
    label: str,
    failure: BaseException | None,
) -> None:
    close_error: OSError | None = None
    try:
        os.close(descriptor)
    except OSError as exc:
        close_error = exc
    if failure is not None:
        if close_error is not None:
            failure.add_note(f"{label} close also failed: {close_error!r}")
        raise failure
    if close_error is not None:
        close_error.add_note(f"while closing {label}")
        raise close_error


def _create_output_file(
    directory_fd: int,
    filename: str,
    *,
    label: str,
) -> tuple[int, tuple[int, int]]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
    identity: tuple[int, int] | None = None
    try:
        opened = os.fstat(descriptor)
        identity = _inode(opened)
        located = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or identity != _inode(located):
            _fail(f"{label} is not one new single-link regular file")
        return descriptor, identity
    except BaseException as failure:
        if identity is None:
            try:
                recovered = os.fstat(descriptor)
                if not stat.S_ISREG(recovered.st_mode):
                    _fail(f"{label} retained creation descriptor is not a regular file")
                identity = _inode(recovered)
            except BaseException as exc:
                failure.add_note(f"{label} identity recovery also failed: {exc!r}")
        if identity is not None:
            try:
                _unlink_owned_file_at(
                    directory_fd,
                    filename,
                    expected_identity=identity,
                    label=label,
                )
            except BaseException as exc:
                failure.add_note(f"{label} partial-create cleanup also failed: {exc!r}")
        try:
            os.close(descriptor)
        except OSError as exc:
            failure.add_note(f"{label} descriptor close also failed: {exc!r}")
        raise


def _unlink_owned_file_at(
    directory_fd: int,
    filename: str,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    located = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(located.st_mode) or _inode(located) != expected_identity:
        _fail(f"{label} partial-create cleanup path was substituted")
    os.unlink(filename, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _finish_output_file(
    descriptor: int,
    directory_fd: int,
    filename: str,
    *,
    identity: tuple[int, int],
    expected_size: int,
    label: str,
) -> None:
    os.fchmod(descriptor, 0o444)
    os.fsync(descriptor)
    opened = os.fstat(descriptor)
    located = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) != 0o444
        or opened.st_size != expected_size
        or _inode(opened) != identity
        or _inode(located) != identity
    ):
        _fail(f"{label} identity or metadata changed while written")


def _write_manifest(
    staging_fd: int,
    raw: bytes,
) -> tuple[int, int]:
    descriptor, identity = _create_output_file(
        staging_fd,
        _MANIFEST_FILENAME,
        label="capture manifest publication file",
    )
    failure: BaseException | None = None
    try:
        _write_all(descriptor, raw, label="capture manifest publication")
        _finish_output_file(
            descriptor,
            staging_fd,
            _MANIFEST_FILENAME,
            identity=identity,
            expected_size=len(raw),
            label="capture manifest publication file",
        )
    except BaseException as exc:
        failure = exc
    try:
        _close_preserving_failure(
            descriptor,
            label="capture manifest publication descriptor",
            failure=failure,
        )
    except BaseException as exc:
        failure = exc
    if failure is not None:
        try:
            _unlink_owned_file_at(
                staging_fd,
                _MANIFEST_FILENAME,
                expected_identity=identity,
                label="capture manifest publication file",
            )
        except BaseException as exc:
            failure.add_note(f"capture manifest partial-create cleanup also failed: {exc!r}")
        raise failure
    return identity


def _response_value(response: HTTPSCaptureResponse, name: str) -> object:
    try:
        return getattr(response, name)
    except BaseException as exc:
        raise ForagerMatchedV3CpuWheelCaptureError(
            f"HTTPS transport response has no stable {name}"
        ) from exc


def _fetch_into_file(
    *,
    transport: HTTPSCaptureTransport,
    url: str,
    timeout_seconds: float,
    output_fd: int,
    expected_size: int,
    expected_sha256: str,
    bytes_before: int,
    global_expected_size: int,
    filename: str,
) -> int:
    try:
        response = transport.open(url, timeout_seconds=timeout_seconds)
    except Exception as exc:
        raise ForagerMatchedV3CpuWheelCaptureError(
            f"HTTPS transport failed to open exact origin for {filename}"
        ) from exc
    failure: BaseException | None = None
    received = 0
    try:
        status = _response_value(response, "status_code")
        final_url = _response_value(response, "final_url")
        if type(status) is not int or status != 200:
            _fail(f"HTTPS status for {filename} is not 200; redirects are rejected")
        if type(final_url) is not str or final_url != url:
            _fail(f"HTTPS redirect or final URL change is forbidden for {filename}")
        digest = hashlib.sha256()
        while True:
            request_size = min(_READ_CHUNK_BYTES, expected_size - received + 1)
            try:
                block = response.read(request_size)
            except Exception as exc:
                raise ForagerMatchedV3CpuWheelCaptureError(
                    f"HTTPS response read failed for {filename}"
                ) from exc
            if type(block) is not bytes:
                _fail(f"HTTPS response returned non-bytes content for {filename}")
            if len(block) > request_size:
                _fail(f"HTTPS transport exceeded the requested read bound for {filename}")
            if not block:
                break
            if received + len(block) > expected_size:
                _fail(f"HTTPS response size exceeds manifest size for {filename}")
            if bytes_before + received + len(block) > global_expected_size:
                _fail("HTTPS response bytes exceed the manifest global byte bound")
            _write_all(output_fd, block, label=f"captured wheel {filename}")
            digest.update(block)
            received += len(block)
        if received != expected_size:
            _fail(f"HTTPS response size differs from manifest size for {filename}")
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            _fail(f"HTTPS response SHA-256 differs from manifest SHA-256 for {filename}")
    except BaseException as exc:
        failure = exc
    try:
        response.close()
    except BaseException as exc:
        if failure is None:
            failure = ForagerMatchedV3CpuWheelCaptureError(
                f"HTTPS response close failed for {filename}"
            )
            failure.__cause__ = exc
        else:
            failure.add_note(f"HTTPS response close also failed: {exc!r}")
    if failure is not None:
        raise failure
    return received


def _hash_open_file(
    descriptor: int,
    size: int,
    *,
    label: str,
) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(_READ_CHUNK_BYTES, size - offset), offset)
        if not block:
            _fail(f"{label} was truncated while hashed")
        digest.update(block)
        offset += len(block)
    if os.pread(descriptor, 1, size):
        _fail(f"{label} exceeds its retained size")
    return digest.hexdigest()


def _validate_file_at(
    directory_fd: int,
    filename: str,
    *,
    expected_identity: tuple[int, int],
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_size != expected_size
        or _inode(before) != expected_identity
    ):
        _fail(f"{label} metadata differs from its retained identity")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(filename, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        observed_sha256 = _hash_open_file(descriptor, expected_size, label=label)
        after = os.fstat(descriptor)
        located = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if (
            _inode(opened) != expected_identity
            or _inode(after) != expected_identity
            or _inode(located) != expected_identity
            or opened.st_size != expected_size
            or after.st_size != expected_size
        ):
            _fail(f"{label} identity changed while validated")
        if not hmac.compare_digest(observed_sha256, expected_sha256):
            _fail(f"{label} SHA-256 differs while validated")
    finally:
        os.close(descriptor)


def _validate_capture_tree(
    staging_fd: int,
    wheels_fd: int,
    *,
    staging_identity: tuple[int, int],
    wheels_identity: tuple[int, int],
    manifest_raw: bytes,
    manifest_sha256: str,
    manifest_identity: tuple[int, int],
    manifest_wheels: list[dict[str, Any]],
    wheel_identities: dict[str, tuple[int, int]],
) -> None:
    staging_metadata = os.fstat(staging_fd)
    wheels_metadata = os.fstat(wheels_fd)
    located_wheels = os.stat(_WHEELS_DIRECTORY, dir_fd=staging_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(staging_metadata.st_mode)
        or stat.S_IMODE(staging_metadata.st_mode) != 0o555
        or _inode(staging_metadata) != staging_identity
        or not stat.S_ISDIR(wheels_metadata.st_mode)
        or stat.S_IMODE(wheels_metadata.st_mode) != 0o555
        or _inode(wheels_metadata) != wheels_identity
        or _inode(located_wheels) != wheels_identity
    ):
        _fail("capture staging directory identities or modes differ")
    if sorted(os.listdir(staging_fd)) != [_MANIFEST_FILENAME, _WHEELS_DIRECTORY]:
        _fail("capture staging directory is not exactly enumerated")
    expected_names = [cast(str, wheel["filename"]) for wheel in manifest_wheels]
    if sorted(os.listdir(wheels_fd)) != expected_names or set(wheel_identities) != set(
        expected_names
    ):
        _fail("captured wheel directory is not exactly enumerated")
    _validate_file_at(
        staging_fd,
        _MANIFEST_FILENAME,
        expected_identity=manifest_identity,
        expected_size=len(manifest_raw),
        expected_sha256=manifest_sha256,
        label="published capture manifest",
    )
    for wheel in manifest_wheels:
        filename = cast(str, wheel["filename"])
        _validate_file_at(
            wheels_fd,
            filename,
            expected_identity=wheel_identities[filename],
            expected_size=cast(int, wheel["size_bytes"]),
            expected_sha256=cast(str, wheel["sha256"]),
            label=f"captured wheel {filename}",
        )


def _assert_publication_bindings(
    *,
    publication_root: Path,
    root_fd: int,
    root_identity: tuple[int, int],
    namespace_fd: int,
    namespace_identity: tuple[int, int],
    staging_fd: int,
    staging_identity: tuple[int, int],
    staging_entry_name: str,
    wheels_fd: int,
    wheels_identity: tuple[int, int],
) -> None:
    try:
        root_named = publication_root.lstat()
        namespace_named = os.stat(_NAMESPACE, dir_fd=root_fd, follow_symlinks=False)
        staging_named = os.stat(
            staging_entry_name,
            dir_fd=namespace_fd,
            follow_symlinks=False,
        )
        wheels_named = os.stat(
            _WHEELS_DIRECTORY,
            dir_fd=staging_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ForagerMatchedV3CpuWheelCaptureError(
            "publication root, namespace, staging, or wheels binding disappeared"
        ) from exc
    if _inode(root_named) != root_identity or _inode(os.fstat(root_fd)) != root_identity:
        _fail("publication root no longer names its retained directory")
    if (
        not stat.S_ISDIR(namespace_named.st_mode)
        or _inode(namespace_named) != namespace_identity
        or _inode(os.fstat(namespace_fd)) != namespace_identity
    ):
        _fail("publication namespace no longer names its retained directory")
    if (
        not stat.S_ISDIR(staging_named.st_mode)
        or _inode(staging_named) != staging_identity
        or _inode(os.fstat(staging_fd)) != staging_identity
    ):
        _fail("capture staging name no longer names its retained directory")
    if (
        not stat.S_ISDIR(wheels_named.st_mode)
        or _inode(wheels_named) != wheels_identity
        or _inode(os.fstat(wheels_fd)) != wheels_identity
    ):
        _fail("capture wheels name no longer names its retained directory")


def _rename_new_only(parent_fd: int, source: str, target: str) -> None:
    renameat2 = cast(Any, getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None))
    if renameat2 is None:
        _fail("atomic new-only capture publication is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(parent_fd, os.fsencode(source), parent_fd, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(target)
    raise OSError(error_number, os.strerror(error_number), target)


def _owned_name(
    parent_fd: int,
    names: tuple[str, ...],
    identity: tuple[int, int],
) -> str | None:
    matches: list[str] = []
    for name in names:
        try:
            located = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(located.st_mode) and _inode(located) == identity:
            matches.append(name)
    if len(matches) > 1:
        _fail("one retained capture directory unexpectedly has multiple names")
    return matches[0] if matches else None


def _rollback_content_address(
    namespace_fd: int,
    digest: str,
    staging_identity: tuple[int, int],
) -> str:
    rollback = f".rollback-{digest}-{os.getpid()}-{secrets.token_hex(16)}"
    try:
        _rename_new_only(namespace_fd, digest, rollback)
    except BaseException:
        located = _owned_name(namespace_fd, (rollback, digest), staging_identity)
        if located == rollback:
            os.fsync(namespace_fd)
            return rollback
        raise
    located = _owned_name(namespace_fd, (rollback, digest), staging_identity)
    if located != rollback:
        _fail("capture rollback no longer names its retained directory")
    os.fsync(namespace_fd)
    return rollback


def _validate_cleanup_file(
    directory_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    located = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(located.st_mode) or _inode(located) != expected_identity:
        _fail(f"{label} cleanup name no longer identifies the created file")


def _cleanup_owned_staging(
    namespace_fd: int,
    entry_name: str,
    staging_fd: int,
    *,
    staging_identity: tuple[int, int],
    wheels_fd: int,
    wheels_identity: tuple[int, int],
    wheel_identities: dict[str, tuple[int, int]],
    manifest_identity: tuple[int, int] | None,
) -> None:
    entry = os.stat(entry_name, dir_fd=namespace_fd, follow_symlinks=False)
    if not stat.S_ISDIR(entry.st_mode) or _inode(entry) != staging_identity:
        _fail("capture staging cleanup path no longer identifies the retained directory")
    staging_names = sorted(os.listdir(staging_fd))
    if wheels_fd < 0:
        if wheel_identities or manifest_identity is not None or staging_names:
            _fail("incomplete capture staging cleanup encountered an unexpected entry")
        os.fchmod(staging_fd, 0o700)
        entry_after = os.stat(entry_name, dir_fd=namespace_fd, follow_symlinks=False)
        if _inode(entry_after) != staging_identity:
            _fail("capture staging cleanup name changed before removal")
        os.rmdir(entry_name, dir_fd=namespace_fd)
        os.fsync(namespace_fd)
        return
    expected_staging = [_WHEELS_DIRECTORY]
    if manifest_identity is not None:
        expected_staging.insert(0, _MANIFEST_FILENAME)
    if staging_names != expected_staging:
        _fail("capture staging cleanup encountered an unexpected entry")
    wheels_named = os.stat(_WHEELS_DIRECTORY, dir_fd=staging_fd, follow_symlinks=False)
    if not stat.S_ISDIR(wheels_named.st_mode) or _inode(wheels_named) != wheels_identity:
        _fail("capture wheels cleanup path no longer identifies the retained directory")
    observed_wheels = sorted(os.listdir(wheels_fd))
    if observed_wheels != sorted(wheel_identities):
        _fail("capture wheels cleanup encountered an unexpected entry")
    for filename in observed_wheels:
        _validate_cleanup_file(
            wheels_fd,
            filename,
            wheel_identities[filename],
            label=f"captured wheel {filename}",
        )
    if manifest_identity is not None:
        _validate_cleanup_file(
            staging_fd,
            _MANIFEST_FILENAME,
            manifest_identity,
            label="capture manifest",
        )
    os.fchmod(wheels_fd, 0o700)
    for filename in observed_wheels:
        os.unlink(filename, dir_fd=wheels_fd)
    os.fsync(wheels_fd)
    if os.listdir(wheels_fd):
        _fail("capture wheels directory is not empty after bounded cleanup")
    os.fchmod(staging_fd, 0o700)
    os.rmdir(_WHEELS_DIRECTORY, dir_fd=staging_fd)
    if manifest_identity is not None:
        os.unlink(_MANIFEST_FILENAME, dir_fd=staging_fd)
    os.fsync(staging_fd)
    if os.listdir(staging_fd):
        _fail("capture staging directory is not empty after bounded cleanup")
    entry_after = os.stat(entry_name, dir_fd=namespace_fd, follow_symlinks=False)
    if _inode(entry_after) != staging_identity:
        _fail("capture staging cleanup name changed before removal")
    os.rmdir(entry_name, dir_fd=namespace_fd)
    os.fsync(namespace_fd)


def _remove_created_namespace(
    root_fd: int,
    namespace_fd: int,
    namespace_identity: tuple[int, int],
) -> None:
    named = os.stat(_NAMESPACE, dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(named.st_mode)
        or _inode(named) != namespace_identity
        or _inode(os.fstat(namespace_fd)) != namespace_identity
    ):
        _fail("publication namespace cleanup path was substituted")
    if os.listdir(namespace_fd):
        _fail("created publication namespace is not empty after cleanup")
    os.rmdir(_NAMESPACE, dir_fd=root_fd)
    os.fsync(root_fd)


def _validate_timeout(value: float) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or not 0.0 < value <= _MAX_REQUEST_TIMEOUT_SECONDS
    ):
        _fail(
            "request timeout must be finite, positive, and no greater than "
            f"{_MAX_REQUEST_TIMEOUT_SECONDS:g} seconds"
        )
    return float(value)


def _validate_production_origins(manifest_wheels: list[dict[str, Any]]) -> None:
    """Constrain untrusted network targets to exact production wheel paths."""

    for wheel in manifest_wheels:
        filename = cast(str, wheel["filename"])
        origin = cast(str, wheel["origin_url"])
        matched = _PYPI_WHEEL_ORIGIN_RE.fullmatch(origin)
        if matched is None or matched.group("filename") != filename:
            _fail(
                f"capture wheel {filename} does not use its exact files.pythonhosted.org "
                "production origin"
            )


def capture_matched_v3_cpu_wheels(
    *,
    capture_manifest_raw: bytes,
    expected_capture_manifest_sha256: str,
    publication_root: Path,
    transport: HTTPSCaptureTransport | None = None,
    authorize_network_capture: bool,
    request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> PublishedMatchedV3CpuWheelCapture:
    """Fetch and atomically publish bytes matching one untrusted manifest.

    ``authorize_network_capture`` must be the literal boolean ``True``.  The
    operation follows zero redirects and publishes no receipt or authority
    statement beyond the original exact manifest bytes.  The flat ``wheels``
    directory can be supplied directly as the disconnected wheelhouse stage's
    ``candidate_directory``.  If descriptor cleanup fails after atomic commit,
    the function reports an error and retains the deterministic new-only
    publication rather than claiming an ordinary successful handoff.
    """

    if authorize_network_capture is not True:
        _fail("network capture requires explicit authorize_network_capture=True authorization")
    timeout = _validate_timeout(request_timeout_seconds)
    manifest = _wheelhouse.parse_cpu_wheel_capture_manifest(
        capture_manifest_raw,
        expected_file_sha256=expected_capture_manifest_sha256,
    )
    manifest_sha256 = hashlib.sha256(capture_manifest_raw).hexdigest()
    manifest_wheels = [cast(dict[str, Any], item) for item in manifest["wheels"]]
    _validate_production_origins(manifest_wheels)
    total_expected_size = sum(cast(int, item["size_bytes"]) for item in manifest_wheels)
    selected_transport: HTTPSCaptureTransport = (
        StdlibHTTPSCaptureTransport() if transport is None else transport
    )

    root_fd = -1
    namespace_fd = -1
    staging_fd = -1
    wheels_fd = -1
    root_identity = (-1, -1)
    namespace_identity = (-1, -1)
    staging_identity = (-1, -1)
    wheels_identity = (-1, -1)
    namespace_created = False
    staging_name: str | None = None
    manifest_identity: tuple[int, int] | None = None
    wheel_identities: dict[str, tuple[int, int]] = {}
    succeeded = False
    failure: BaseException | None = None
    collision: FileExistsError | None = None
    cleanup_errors: list[BaseException] = []

    try:
        root_fd, root_identity = _open_absolute_directory(
            publication_root,
            label="CPU wheel capture publication root",
        )
        try:
            namespace_fd, namespace_identity = _create_directory_at(
                root_fd,
                _NAMESPACE,
                label="CPU wheel capture publication namespace",
            )
            namespace_created = True
            os.fsync(root_fd)
        except FileExistsError:
            namespace_fd, namespace_identity = _open_directory_at(
                root_fd,
                _NAMESPACE,
                label="CPU wheel capture publication namespace",
            )
        try:
            os.stat(manifest_sha256, dir_fd=namespace_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(manifest_sha256)

        staging_name = f".capture-{manifest_sha256}-{os.getpid()}-{secrets.token_hex(16)}"
        staging_fd, staging_identity = _create_directory_at(
            namespace_fd,
            staging_name,
            label="private CPU wheel capture staging directory",
        )
        wheels_fd, wheels_identity = _create_directory_at(
            staging_fd,
            _WHEELS_DIRECTORY,
            label="private CPU wheel capture wheels directory",
        )

        received_total = 0
        for wheel in manifest_wheels:
            filename = cast(str, wheel["filename"])
            output_fd, output_identity = _create_output_file(
                wheels_fd,
                filename,
                label=f"captured wheel {filename}",
            )
            wheel_identities[filename] = output_identity
            output_failure: BaseException | None = None
            received = 0
            try:
                received = _fetch_into_file(
                    transport=selected_transport,
                    url=cast(str, wheel["origin_url"]),
                    timeout_seconds=timeout,
                    output_fd=output_fd,
                    expected_size=cast(int, wheel["size_bytes"]),
                    expected_sha256=cast(str, wheel["sha256"]),
                    bytes_before=received_total,
                    global_expected_size=total_expected_size,
                    filename=filename,
                )
                _finish_output_file(
                    output_fd,
                    wheels_fd,
                    filename,
                    identity=output_identity,
                    expected_size=cast(int, wheel["size_bytes"]),
                    label=f"captured wheel {filename}",
                )
                received_total += received
            except BaseException as exc:
                output_failure = exc
            _close_preserving_failure(
                output_fd,
                label=f"captured wheel {filename} descriptor",
                failure=output_failure,
            )
        if received_total != total_expected_size:
            _fail("captured wheels differ from the manifest global byte total")

        manifest_identity = _write_manifest(staging_fd, capture_manifest_raw)
        os.fsync(wheels_fd)
        os.fchmod(wheels_fd, 0o555)
        os.fsync(wheels_fd)
        os.fsync(staging_fd)
        os.fchmod(staging_fd, 0o555)
        os.fsync(staging_fd)
        _assert_publication_bindings(
            publication_root=publication_root,
            root_fd=root_fd,
            root_identity=root_identity,
            namespace_fd=namespace_fd,
            namespace_identity=namespace_identity,
            staging_fd=staging_fd,
            staging_identity=staging_identity,
            staging_entry_name=staging_name,
            wheels_fd=wheels_fd,
            wheels_identity=wheels_identity,
        )
        _validate_capture_tree(
            staging_fd,
            wheels_fd,
            staging_identity=staging_identity,
            wheels_identity=wheels_identity,
            manifest_raw=capture_manifest_raw,
            manifest_sha256=manifest_sha256,
            manifest_identity=manifest_identity,
            manifest_wheels=manifest_wheels,
            wheel_identities=wheel_identities,
        )
        os.fsync(namespace_fd)
        _rename_new_only(namespace_fd, staging_name, manifest_sha256)
        os.fsync(namespace_fd)
        _validate_capture_tree(
            staging_fd,
            wheels_fd,
            staging_identity=staging_identity,
            wheels_identity=wheels_identity,
            manifest_raw=capture_manifest_raw,
            manifest_sha256=manifest_sha256,
            manifest_identity=manifest_identity,
            manifest_wheels=manifest_wheels,
            wheel_identities=wheel_identities,
        )
        _assert_publication_bindings(
            publication_root=publication_root,
            root_fd=root_fd,
            root_identity=root_identity,
            namespace_fd=namespace_fd,
            namespace_identity=namespace_identity,
            staging_fd=staging_fd,
            staging_identity=staging_identity,
            staging_entry_name=manifest_sha256,
            wheels_fd=wheels_fd,
            wheels_identity=wheels_identity,
        )
        succeeded = True
    except FileExistsError as exc:
        collision = exc
    except BaseException as exc:
        failure = exc

    if not succeeded and staging_fd >= 0 and namespace_fd >= 0 and staging_name is not None:
        try:
            owned_entry = _owned_name(
                namespace_fd,
                (staging_name, manifest_sha256),
                staging_identity,
            )
            if owned_entry == manifest_sha256:
                owned_entry = _rollback_content_address(
                    namespace_fd,
                    manifest_sha256,
                    staging_identity,
                )
            if owned_entry is None:
                _fail("capture staging cleanup name was substituted or removed")
            _cleanup_owned_staging(
                namespace_fd,
                owned_entry,
                staging_fd,
                staging_identity=staging_identity,
                wheels_fd=wheels_fd,
                wheels_identity=wheels_identity,
                wheel_identities=wheel_identities,
                manifest_identity=manifest_identity,
            )
        except BaseException as exc:
            cleanup_errors.append(exc)
    if not succeeded and namespace_created and root_fd >= 0 and namespace_fd >= 0:
        try:
            _remove_created_namespace(root_fd, namespace_fd, namespace_identity)
        except BaseException as exc:
            cleanup_errors.append(exc)

    for descriptor, label in (
        (wheels_fd, "capture wheels directory descriptor"),
        (staging_fd, "capture staging directory descriptor"),
        (namespace_fd, "capture namespace descriptor"),
        (root_fd, "capture publication-root descriptor"),
    ):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                exc.add_note(f"while closing {label}")
                cleanup_errors.append(exc)

    if collision is not None:
        wrapped = FileExistsError(
            f"refusing to overwrite CPU wheel capture publication {manifest_sha256}"
        )
        for cleanup_error in cleanup_errors:
            wrapped.add_note(f"cleanup also failed: {cleanup_error!r}")
        raise wrapped from collision
    if failure is not None:
        for cleanup_error in cleanup_errors:
            failure.add_note(f"cleanup also failed: {cleanup_error!r}")
        raise failure
    if succeeded and cleanup_errors:
        committed_error = ForagerMatchedV3CpuWheelCaptureError(
            f"CPU wheel capture publication {manifest_sha256} was committed but descriptor "
            "cleanup failed"
        )
        for cleanup_error in cleanup_errors:
            committed_error.add_note(f"descriptor cleanup failed: {cleanup_error!r}")
        raise committed_error from cleanup_errors[0]
    if not succeeded:
        if cleanup_errors:
            primary = cleanup_errors[0]
            for cleanup_error in cleanup_errors[1:]:
                primary.add_note(f"cleanup also failed: {cleanup_error!r}")
            raise primary
        _fail("CPU wheel capture stopped without a result")

    directory = publication_root / _NAMESPACE / manifest_sha256
    return PublishedMatchedV3CpuWheelCapture(
        directory=directory,
        manifest=directory / _MANIFEST_FILENAME,
        wheels=directory / _WHEELS_DIRECTORY,
        manifest_sha256=manifest_sha256,
        wheel_count=len(manifest_wheels),
        total_size_bytes=total_expected_size,
    )


__all__ = [
    "CPU_WHEEL_CAPTURE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "CPU_WHEEL_CAPTURE_STATUS",
    "ForagerMatchedV3CpuWheelCaptureError",
    "HTTPSCaptureResponse",
    "HTTPSCaptureTransport",
    "PublishedMatchedV3CpuWheelCapture",
    "StdlibHTTPSCaptureTransport",
    "capture_matched_v3_cpu_wheels",
    "cpu_wheel_capture_contract_descriptor",
]
