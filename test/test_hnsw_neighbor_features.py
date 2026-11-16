import unittest
from datetime import datetime

import numpy as np
import pytest

from datasketch.hnsw_neighbor_features import (
    NeighborFeatures,
    PredictionRow,
    ReferenceRecord,
    TargetEvent,
    TargetEventType,
    construct_hnsw_neighbor_features,
)


def l2_distance(x, y):
    return float(np.linalg.norm(x - y))


class TestHNSWNeighborFeatures(unittest.TestCase):
    def _make_records(self):
        # Four reference records on a 1-D line for deterministic ordering.
        return [
            ReferenceRecord(record_id="r0", entity_id="a", as_of=1, point=np.array([0.0])),
            ReferenceRecord(record_id="r1", entity_id="b", as_of=1, point=np.array([1.0])),
            ReferenceRecord(record_id="r2", entity_id="b", as_of=2, point=np.array([2.0])),
            ReferenceRecord(record_id="r3", entity_id="c", as_of=2, point=np.array([3.0])),
        ]

    def _publish_all(self, records, value_by_id=None):
        if value_by_id is None:
            # Target value equals the point coordinate for easy math.
            value_by_id = {record.record_id: float(record.point[0]) for record in records}
        return [
            TargetEvent(
                record_id=record.record_id,
                timestamp=record.as_of,
                event_type=TargetEventType.PUBLISH,
                value=value_by_id[record.record_id],
            )
            for record in records
        ]

    def _construct(self, rows, records, events, k=10, **kwargs):
        return construct_hnsw_neighbor_features(
            rows,
            records,
            events,
            distance_func=l2_distance,
            k=k,
            seed=42,
            **kwargs,
        )

    def test_basic_features_match_brute_force(self):
        records = self._make_records()
        events = self._publish_all(records)
        row = PredictionRow(row_id="p0", entity_id="d", as_of=3, point=np.array([1.4]))
        results = self._construct([row], records, events, k=3)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsInstance(result, NeighborFeatures)
        self.assertEqual(result.row_id, "p0")
        # All four records are earlier, different entity, and published.
        self.assertEqual(result.n_neighbors, 3)
        # Nearest order on the line: r1 (1.0), r2 (2.0), r0 (0.0).
        self.assertEqual(result.neighbor_ids, ("r1", "r2", "r0"))
        self.assertEqual(len(result.neighbor_distances), 3)
        for reported, expected in zip(result.neighbor_distances, (0.4, 0.6, 1.4)):
            self.assertAlmostEqual(reported, expected)
        self.assertAlmostEqual(result.nearest_distance, 0.4)
        self.assertAlmostEqual(result.average_distance, (0.4 + 0.6 + 1.4) / 3)
        self.assertAlmostEqual(result.target_mean, (1.0 + 2.0 + 0.0) / 3)
        # Citations and distances are aligned.
        self.assertEqual(len(result.neighbor_ids), len(result.neighbor_distances))

    def test_k_caps_neighbor_count(self):
        records = self._make_records()
        events = self._publish_all(records)
        row = PredictionRow(row_id="p0", entity_id="d", as_of=3, point=np.array([1.4]))
        result = self._construct([row], records, events, k=2)[0]
        self.assertEqual(result.n_neighbors, 2)
        self.assertEqual(result.neighbor_ids, ("r1", "r2"))

    def test_strictly_earlier_timestamp_excludes_future_and_same_time_records(self):
        records = self._make_records()
        events = self._publish_all(records)
        # as_of=2: records at as_of=2 (r2, r3) are not strictly earlier.
        row = PredictionRow(row_id="p0", entity_id="d", as_of=2, point=np.array([0.5]))
        result = self._construct([row], records, events, k=10)[0]
        self.assertEqual(result.n_neighbors, 2)
        self.assertEqual(set(result.neighbor_ids), {"r0", "r1"})

        # as_of=1: no record is strictly earlier -> empty population.
        row_early = PredictionRow(row_id="p1", entity_id="d", as_of=1, point=np.array([0.5]))
        result_early = self._construct([row_early], records, events)[0]
        self._assert_empty_population(result_early, "p1")

    def test_same_entity_records_are_excluded(self):
        records = self._make_records()
        events = self._publish_all(records)
        # Row belongs to entity "b": r1 and r2 must be excluded.
        row = PredictionRow(row_id="p0", entity_id="b", as_of=3, point=np.array([2.1]))
        result = self._construct([row], records, events, k=10)[0]
        self.assertEqual(set(result.neighbor_ids), {"r0", "r3"})
        for cited_id in result.neighbor_ids:
            self.assertNotEqual(records_by_id(records, cited_id).entity_id, "b")

    def test_target_not_yet_published_excludes_record(self):
        records = self._make_records()
        events = self._publish_all(records)
        # Delay r1's publication until timestamp 4.
        events = [event for event in events if event.record_id != "r1"]
        events.append(TargetEvent(record_id="r1", timestamp=4, event_type=TargetEventType.PUBLISH, value=1.0))
        row = PredictionRow(row_id="p0", entity_id="d", as_of=3, point=np.array([1.0]))
        result = self._construct([row], records, events, k=10)[0]
        self.assertNotIn("r1", result.neighbor_ids)
        self.assertEqual(set(result.neighbor_ids), {"r0", "r2", "r3"})

        # At as_of=4 the delayed publication is visible (event timestamp <= as_of).
        row_later = PredictionRow(row_id="p1", entity_id="d", as_of=4, point=np.array([1.0]))
        result_later = self._construct([row_later], records, events, k=10)[0]
        self.assertIn("r1", result_later.neighbor_ids)

    def test_retraction_removes_record_until_republished(self):
        records = self._make_records()
        events = self._publish_all(records)
        # r1 published at 1, retracted at 2, republished at 4 with a new value.
        events.append(TargetEvent(record_id="r1", timestamp=2, event_type=TargetEventType.RETRACT))
        events.append(TargetEvent(record_id="r1", timestamp=4, event_type=TargetEventType.PUBLISH, value=99.0))
        row = PredictionRow(row_id="p0", entity_id="d", as_of=3, point=np.array([1.0]))
        result = self._construct([row], records, events, k=10)[0]
        self.assertNotIn("r1", result.neighbor_ids)

        row_republished = PredictionRow(row_id="p1", entity_id="d", as_of=4, point=np.array([1.0]))
        result_republished = self._construct([row_republished], records, events, k=10)[0]
        self.assertIn("r1", result_republished.neighbor_ids)
        # The republished value (99.0) feeds the mean, not the original 1.0.
        other_ids = [rid for rid in result_republished.neighbor_ids if rid != "r1"]
        other_target_sum = sum(float(records_by_id(records, rid).point[0]) for rid in other_ids)
        r1_target = result_republished.target_mean * result_republished.n_neighbors - other_target_sum
        self.assertAlmostEqual(r1_target, 99.0)

    def test_revision_uses_latest_published_value_known_at_as_of(self):
        records = self._make_records()
        events = self._publish_all(records)
        # r0 is revised at timestamp 2 from 0.0 to 10.0, and again at timestamp 5.
        events.append(TargetEvent(record_id="r0", timestamp=2, event_type=TargetEventType.PUBLISH, value=10.0))
        events.append(TargetEvent(record_id="r0", timestamp=5, event_type=TargetEventType.PUBLISH, value=50.0))
        row = PredictionRow(row_id="p0", entity_id="d", as_of=3, point=np.array([0.0]))
        result = self._construct([row], records, events, k=1)[0]
        self.assertEqual(result.neighbor_ids, ("r0",))
        # Mean uses the value known at as_of=3, i.e. the 10.0 revision.
        self.assertAlmostEqual(result.target_mean, 10.0)

        # A future revision (timestamp 5) must not leak into an as_of=3 row.
        row2 = PredictionRow(row_id="p1", entity_id="d", as_of=5, point=np.array([0.0]))
        result2 = self._construct([row2], records, events, k=1)[0]
        self.assertAlmostEqual(result2.target_mean, 50.0)

    def test_record_without_any_target_event_is_excluded(self):
        records = self._make_records()
        events = [event for event in self._publish_all(records) if event.record_id != "r3"]
        row = PredictionRow(row_id="p0", entity_id="d", as_of=3, point=np.array([3.0]))
        result = self._construct([row], records, events, k=10)[0]
        self.assertNotIn("r3", result.neighbor_ids)

    def _assert_empty_population(self, result, row_id):
        self.assertEqual(result.row_id, row_id)
        self.assertEqual(result.n_neighbors, 0)
        self.assertIsNone(result.target_mean)
        self.assertIsNone(result.nearest_distance)
        self.assertIsNone(result.average_distance)
        self.assertEqual(result.neighbor_ids, ())
        self.assertEqual(result.neighbor_distances, ())

    def test_empty_population_retains_row_with_missing_values(self):
        # A dedicated dataset where each empty-population cause is unambiguous.
        records = [
            ReferenceRecord(record_id="e0", entity_id="a", as_of=1, point=np.array([0.0])),
            ReferenceRecord(record_id="e1", entity_id="a", as_of=2, point=np.array([1.0])),
        ]
        rows = [
            # No strictly earlier records.
            PredictionRow(row_id="early", entity_id="b", as_of=1, point=np.array([0.5])),
            # Every earlier record shares the entity.
            PredictionRow(row_id="same_entity", entity_id="a", as_of=2, point=np.array([0.5])),
            # The only earlier record has an unpublished target at as_of.
            PredictionRow(row_id="unpublished", entity_id="b", as_of=2, point=np.array([0.5])),
        ]
        late_events = [
            TargetEvent(record_id="e0", timestamp=3, event_type=TargetEventType.PUBLISH, value=1.0),
            TargetEvent(record_id="e1", timestamp=3, event_type=TargetEventType.PUBLISH, value=2.0),
        ]
        results = self._construct(rows, records, late_events, k=10)
        self.assertEqual(
            [result.row_id for result in results],
            ["early", "same_entity", "unpublished"],
        )
        for result, row_id in zip(results, ("early", "same_entity", "unpublished")):
            self._assert_empty_population(result, row_id)

    def test_results_aligned_with_input_rows(self):
        records = self._make_records()
        events = self._publish_all(records)
        rows = [PredictionRow(row_id=f"p{i}", entity_id="d", as_of=3, point=np.array([float(i)])) for i in range(5)]
        results = self._construct(rows, records, events)
        self.assertEqual(len(results), len(rows))
        self.assertEqual([result.row_id for result in results], [row.row_id for row in rows])

    def test_rows_sharing_as_of_and_entity_reuse_population(self):
        records = self._make_records()
        events = self._publish_all(records)
        # Two rows with the same (as_of, entity_id) must see the same
        # population regardless of query point.
        rows = [
            PredictionRow(row_id="p0", entity_id="b", as_of=3, point=np.array([0.0])),
            PredictionRow(row_id="p1", entity_id="b", as_of=3, point=np.array([9.0])),
        ]
        results = self._construct(rows, records, events, k=10)
        self.assertEqual(
            set(results[0].neighbor_ids),
            set(results[1].neighbor_ids),
        )
        self.assertEqual(results[0].n_neighbors, results[1].n_neighbors)

    def test_prediction_rows_never_enter_reference_index(self):
        records = self._make_records()
        events = self._publish_all(records)
        row = PredictionRow(row_id="p0", entity_id="d", as_of=3, point=np.array([1.0]))
        # A prediction row whose row_id collides with a reference ID still
        # cannot appear as its own neighbor.
        colliding_row = PredictionRow(row_id="r1", entity_id="d", as_of=3, point=np.array([1.0]))
        results = self._construct([row, colliding_row], records, events, k=10)
        for result in results:
            self.assertNotIn("p0", result.neighbor_ids)
        # r1 remains a legitimate reference for the row of a different entity.
        self.assertIn("r1", results[0].neighbor_ids)

    def test_datetime_timestamps_are_supported(self):
        records = [
            ReferenceRecord(
                record_id="r0",
                entity_id="a",
                as_of=datetime(2024, 1, 1),
                point=np.array([0.0]),
            ),
            ReferenceRecord(
                record_id="r1",
                entity_id="b",
                as_of=datetime(2024, 1, 2),
                point=np.array([1.0]),
            ),
        ]
        events = [
            TargetEvent(
                record_id="r0",
                timestamp=datetime(2024, 1, 1),
                event_type=TargetEventType.PUBLISH,
                value=5.0,
            ),
            TargetEvent(
                record_id="r1",
                timestamp=datetime(2024, 1, 3),
                event_type=TargetEventType.PUBLISH,
                value=7.0,
            ),
        ]
        # At Jan 2, r1 exists as a record but its target is not published yet.
        row = PredictionRow(row_id="p0", entity_id="c", as_of=datetime(2024, 1, 2), point=np.array([0.5]))
        result = self._construct([row], records, events)[0]
        self.assertEqual(result.neighbor_ids, ("r0",))
        self.assertAlmostEqual(result.target_mean, 5.0)

    def test_event_at_exactly_as_of_is_visible(self):
        records = self._make_records()
        # Records r0/r1 exist at as_of=1 but their targets are only published
        # exactly at the row's as_of=2: those events must still be visible.
        events = [
            TargetEvent(record_id="r0", timestamp=2, event_type=TargetEventType.PUBLISH, value=8.0),
            TargetEvent(record_id="r1", timestamp=2, event_type=TargetEventType.PUBLISH, value=9.0),
        ]
        row = PredictionRow(row_id="p0", entity_id="d", as_of=2, point=np.array([2.0]))
        result = self._construct([row], records, events, k=10)[0]
        # r2/r3 are excluded because their records are not strictly earlier.
        self.assertEqual(set(result.neighbor_ids), {"r0", "r1"})
        self.assertAlmostEqual(result.target_mean, 8.5)

    def test_invalid_k_raises(self):
        records = self._make_records()
        events = self._publish_all(records)
        row = PredictionRow(row_id="p0", entity_id="d", as_of=3, point=np.array([1.0]))
        with self.assertRaises(ValueError):
            self._construct([row], records, events, k=0)
        with self.assertRaises(ValueError):
            self._construct([row], records, events, k=-1)

    def test_duplicate_reference_id_raises(self):
        records = self._make_records()
        records.append(ReferenceRecord(record_id="r0", entity_id="z", as_of=5, point=np.array([9.0])))
        events = self._publish_all(self._make_records())
        row = PredictionRow(row_id="p0", entity_id="d", as_of=6, point=np.array([1.0]))
        with self.assertRaises(ValueError):
            self._construct([row], records, events)

    def test_target_event_for_unknown_record_raises(self):
        records = self._make_records()
        events = self._publish_all(records)
        events.append(TargetEvent(record_id="missing", timestamp=1, event_type=TargetEventType.PUBLISH, value=1.0))
        row = PredictionRow(row_id="p0", entity_id="d", as_of=3, point=np.array([1.0]))
        with self.assertRaises(ValueError):
            self._construct([row], records, events)

    def test_publish_event_without_value_raises(self):
        records = self._make_records()
        events = self._publish_all(records)
        events.append(TargetEvent(record_id="r0", timestamp=3, event_type=TargetEventType.PUBLISH))
        row = PredictionRow(row_id="p0", entity_id="d", as_of=4, point=np.array([1.0]))
        with self.assertRaises(ValueError):
            self._construct([row], records, events)

    def test_larger_population_matches_brute_force_membership_and_stats(self):
        rng = np.random.RandomState(0)
        n_records = 60
        records = [
            ReferenceRecord(
                record_id=i,
                entity_id=rng.randint(0, 5),
                as_of=int(rng.randint(1, 10)),
                point=rng.random_sample(4),
            )
            for i in range(n_records)
        ]
        events = [
            TargetEvent(
                record_id=i,
                timestamp=records[i].as_of,
                event_type=TargetEventType.PUBLISH,
                value=float(i),
            )
            for i in range(n_records)
        ]
        row = PredictionRow(row_id="p0", entity_id=99, as_of=10, point=rng.random_sample(4))
        k = 5
        results = self._construct([row], records, events, k=k, ef_construction=200, query_ef=200)
        result = results[0]

        # Compute the exact eligible population and exact kNN.
        eligible = {
            record.record_id: record
            for record in records
            if record.as_of < row.as_of and record.entity_id != row.entity_id
        }
        exact = sorted(
            ((record_id, l2_distance(row.point, record.point)) for record_id, record in eligible.items()),
            key=lambda item: item[1],
        )[:k]
        exact_ids = [record_id for record_id, _ in exact]
        exact_values = [float(record_id) for record_id in exact_ids]

        self.assertEqual(result.n_neighbors, len(exact))
        # With a small, well-connected index and high ef, HNSW returns the
        # exact nearest neighbors.
        self.assertEqual(list(result.neighbor_ids), exact_ids)
        self.assertEqual(len(result.neighbor_distances), k)
        # Distances are sorted ascending and match exact distances.
        for reported, (_, expected_distance) in zip(result.neighbor_distances, exact):
            self.assertAlmostEqual(reported, expected_distance)
        self.assertAlmostEqual(result.nearest_distance, exact[0][1])
        self.assertAlmostEqual(
            result.average_distance,
            sum(distance for _, distance in exact) / len(exact),
        )
        self.assertAlmostEqual(result.target_mean, sum(exact_values) / len(exact_values))

    def test_neighbor_count_capped_by_population_size(self):
        records = self._make_records()
        events = self._publish_all(records)
        # Only r0 is eligible (entity b excludes r1/r2, and ask for as_of=2
        # which excludes r2/r3 on timestamp anyway).
        row = PredictionRow(row_id="p0", entity_id="b", as_of=2, point=np.array([0.0]))
        result = self._construct([row], records, events, k=10)[0]
        self.assertEqual(result.n_neighbors, 1)
        self.assertEqual(result.neighbor_ids, ("r0",))


def records_by_id(records, record_id):
    for record in records:
        if record.record_id == record_id:
            return record
    raise KeyError(record_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
