"""Segment-aware walk-forward: fold windows never cross a segment boundary."""

from datetime import date, datetime, timezone

from models.train import Segment, walk_forward_splits


def _us(y: int, m: int, d: int, h: int = 0) -> int:
    return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp() * 1_000_000)


ALL_DATES = [date(2026, 3, d) for d in range(2, 17)]  # Mar 2 .. Mar 16


def test_segment_dates_include_boundary_dates():
    seg_a = Segment("A", _us(2026, 3, 2, 20), _us(2026, 3, 6, 9))
    assert seg_a.dates(ALL_DATES) == [date(2026, 3, d) for d in (2, 3, 4, 5, 6)]


def test_segment_dates_end_exclusive_at_midnight():
    seg = Segment("X", _us(2026, 3, 2), _us(2026, 3, 5))
    assert seg.dates(ALL_DATES)[-1] == date(2026, 3, 4)


def test_folds_stay_inside_segment():
    seg_b = Segment("B", _us(2026, 3, 6, 10), _us(2026, 3, 13, 19))
    seg_dates = seg_b.dates(ALL_DATES)  # Mar 6 .. Mar 13 → 8 dates
    splits = list(walk_forward_splits(seg_dates, min_train_days=3, embargo_days=1))
    assert len(splits) == 4
    for s in splits:
        for d in s.train_dates + s.embargo_dates + [s.test_date]:
            assert date(2026, 3, 6) <= d <= date(2026, 3, 13)


def test_too_few_dates_yields_no_folds():
    seg_c = Segment("C", _us(2026, 3, 13, 20), _us(2026, 3, 16, 22))
    seg_dates = seg_c.dates(ALL_DATES)  # Mar 13 .. Mar 16 → 4 dates
    splits = list(walk_forward_splits(seg_dates, min_train_days=3, embargo_days=1))
    assert splits == []
