from harness.guards import verify_frozen_files


def test_frozen_hashes_match_checkout():
    verify_frozen_files()
