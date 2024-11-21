import pytest

from src.data.buffer import ReplayBuffer


# Test the ReplayBuffer class
@pytest.fixture
def buffer():
    """Fixture to initialize a ReplayBuffer with fixed parameters."""
    recency_temperature = 1.0
    max_size = 5
    return ReplayBuffer(recency_temperature, max_size)


def test_add_and_retrieve_single(buffer):
    """Test adding a single row and retrieving it."""
    row = (1, 2, 3, 4)
    retrieved, remaining = buffer.retrieve(1, row)
    assert len(buffer) == 1, "Buffer should contain one element."
    assert retrieved == [
        (1,),
        (2,),
        (3,),
        (4,),
    ], "Retrieved should contain the transposed single row."
    assert remaining == (
        None,
        None,
        None,
        None,
    ), "Remaining should be empty and transposed to None."


def test_add_and_retrieve_multiple(buffer):
    """Test adding multiple rows and retrieving them."""
    rows = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
    for row in rows:
        buffer.retrieve(0, row)
    retrieved, remaining = buffer.retrieve(2, (13, 14, 15, 16))

    # Validate sizes
    assert len(buffer) == 4, "Buffer should contain four elements after additions."
    assert (
        len(retrieved[0]) == 2
    ), "Retrieved should contain two rows (including the most recent one)."

    # Check most recent row is included
    most_recent = (13, 14, 15, 16)
    assert most_recent in zip(
        *retrieved
    ), "Most recent row should always be included in retrieved."


def test_buffer_max_size(buffer):
    """Test that the buffer does not exceed its maximum size."""
    for i in range(10):
        buffer.retrieve(0, (i, i + 1, i + 2, i + 3))
    assert len(buffer) == buffer.max_size, "Buffer should not exceed max size."

    # Validate recent and old rows
    retrieved, _ = buffer.retrieve(3, (10, 11, 12, 13))
    assert len(retrieved[0]) == 3, "Retrieved should contain three rows."
    assert (10, 11, 12, 13) in zip(*retrieved), "Most recent row should be included."


def test_transposed_empty_remaining(buffer):
    """Test the output format when remaining is empty."""
    # row = (1, 2, 3, 4)
    # buffer.retrieve(1, row)
    retrieved, remaining = buffer.retrieve(1, (5, 6, 7, 8))
    assert remaining == (
        None,
        None,
        None,
        None,
    ), "Remaining should return transposed None when empty."


def test_transposed_output_format(buffer):
    """Test that retrieved and remaining are correctly transposed."""
    rows = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
    for row in rows:
        buffer.retrieve(0, row)
    retrieved, remaining = buffer.retrieve(2, (13, 14, 15, 16))

    # Validate transposition of retrieved
    assert len(retrieved) == 4, "Retrieved should have 4 transposed lists (columns)."
    assert all(
        isinstance(col, list) for col in retrieved
    ), "Each column in retrieved should be a list."

    # Validate transposition of remaining
    if remaining != (None, None, None, None):
        assert (
            len(remaining) == 4
        ), "Remaining should have 4 transposed lists (columns)."
        assert all(
            isinstance(col, list) for col in remaining
        ), "Each column in remaining should be a list."


def test_random_sampling(buffer):
    """Test the random sampling and ensure it respects weights."""
    rows = [(i, i + 1, i + 2, i + 3) for i in range(5)]
    for row in rows:
        buffer.retrieve(0, row)
    retrieved, _ = buffer.retrieve(3, (10, 11, 12, 13))

    # Check that most recent row is always included
    assert (10, 11, 12, 13) in zip(
        *retrieved
    ), "Most recent row should always be included in retrieved."

    # Validate no duplicates in retrieved
    flattened_retrieved = list(zip(*retrieved))
    assert len(flattened_retrieved) == len(
        set(flattened_retrieved)
    ), "Retrieved rows should not contain duplicates."
