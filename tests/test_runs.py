"""The manifest hash helper."""

import hashlib

from traffic_signs.utils.runs import file_sha256


def test_file_sha256_hashes_file_bytes(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_bytes(b'{"version": 1}\n')
    assert file_sha256(path) == hashlib.sha256(b'{"version": 1}\n').hexdigest()
