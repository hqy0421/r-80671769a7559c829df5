"""Point-in-time construction of historical-neighbor features on top of
:class:`~datasketch.hnsw.HNSW`.

This module augments prediction rows with features computed from historically
similar reference records (e.g. the mean target value of the nearest historical
cases). The reference population and the publication history of target values
are kept separate so that information that was not available at a prediction
row's point in time can never leak into its features:

* Only reference records whose ``as_of`` timestamp is **strictly earlier** than
  the prediction row's timestamp are eligible.
* Only reference records belonging to a **different entity** than the
  prediction row are eligible.
* A reference record is eligible only if its target had a **published value at
  the prediction row's timestamp** and the latest known target event was not a
  retraction. Target events published, revised, or withdrawn at or after the
  prediction row's timestamp are invisible to that row.

A fresh :class:`~datasketch.hnsw.HNSW` index is built from the eligible
reference population of each distinct ``(as_of, entity_id)`` combination. This
module does not perform incremental index construction: it relies on the
native approximate nearest neighbor semantics of :class:`~datasketch.hnsw.HNSW`
and never inserts prediction rows into any reference index.

Example:
    Build historical-neighbor features for two prediction rows from reference
    records with a separately tracked target publication history.

    .. code-block:: python

        import numpy as np

        from datasketch.hnsw_neighbor_features import (
            TargetEvent,
            TargetEventType,
            ReferenceRecord,
            PredictionRow,
            construct_hnsw_neighbor_features,
        )

        reference_records = [
            ReferenceRecord(record_id="r1", entity_id="a", as_of=1, point=np.array([0.0, 0.0])),
            ReferenceRecord(record_id="r2", entity_id="b", as_of=1, point=np.array([0.1, 0.1])),
            ReferenceRecord(record_id="r3", entity_id="b", as_of=3, point=np.array([0.2, 0.2])),
        ]
        target_events = [
            TargetEvent(record_id="r1", timestamp=2, event_type=TargetEventType.PUBLISH, value=10.0),
            TargetEvent(record_id="r2", timestamp=2, event_type=TargetEventType.PUBLISH, value=20.0),
            TargetEvent(record_id="r3", timestamp=4, event_type=TargetEventType.PUBLISH, value=30.0),
        ]
        prediction_rows = [
            PredictionRow(row_id="p1", entity_id="a", as_of=3, point=np.array([0.05, 0.05])),
            PredictionRow(row_id="p2", entity_id="a", as_of=3, point=np.array([0.15, 0.15])),
        ]
        results = construct_hnsw_neighbor_features(
            prediction_rows,
            reference_records,
            target_events,
            distance_func=lambda x, y: np.linalg.norm(x - y),
            k=2,
        )

"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from datasketch.hnsw import HNSW


class TargetEventType(Enum):
    """The type of an event in a target value's publication history.

    Attributes:
        PUBLISH: A target value becomes known. A later publish event acts as a
            revision: the most recently published value supersedes earlier ones.
        RETRACT: The previously published target value is withdrawn. A record
            whose latest known event is a retraction has no usable target value
            until a later publish event restores one.

    """

    PUBLISH = "publish"
    RETRACT = "retract"


@dataclass(frozen=True)
class TargetEvent:
    """An event in the publication history of a reference record's target value.

    Attributes:
        record_id: The :attr:`ReferenceRecord.record_id` of the reference
            record whose target this event concerns.
        timestamp: The publication timestamp of the event. Timestamps of the
            same record must be totally ordered and comparable with the
            ``as_of`` timestamps of prediction rows (e.g. integers or
            :class:`datetime.datetime`).
        event_type: Whether the event publishes (or revises) a target value or
            retracts the previously published value.
        value: The published target value. Required for
            :attr:`TargetEventType.PUBLISH` events and ignored for
            :attr:`TargetEventType.RETRACT` events.

    """

    record_id: Hashable
    timestamp: object
    event_type: TargetEventType
    value: Optional[float] = None


@dataclass(frozen=True)
class ReferenceRecord:
    """A record that may serve as a historical reference for prediction rows.

    Attributes:
        record_id: The unique identifier of the record. Referenced by target
            events and returned as a neighbor citation.
        entity_id: The identifier of the entity the record belongs to. Records
            of the same entity as a prediction row are never used as its
            references.
        as_of: The timestamp at which the record's feature point became
            available. Only records with a timestamp strictly earlier than a
            prediction row's timestamp are eligible for that row.
        point: The feature vector indexed by
            :class:`~datasketch.hnsw.HNSW`.

    """

    record_id: Hashable
    entity_id: Hashable
    as_of: object
    point: np.ndarray


@dataclass(frozen=True)
class PredictionRow:
    """A prediction row for which historical-neighbor features are built.

    Prediction rows are never inserted into a reference index; they are only
    used as query points.

    Attributes:
        row_id: The unique identifier of the prediction row, echoed back on the
            result to keep the output aligned with the input.
        entity_id: The identifier of the entity the prediction row belongs to.
            Reference records of the same entity are excluded from its
            neighbors.
        as_of: The prediction timestamp. A reference record or target event is
            visible to this row only if it was known at this timestamp (see
            :mod:`~datasketch.hnsw_neighbor_features` for the exact rules).
        point: The feature vector used as the HNSW query point.

    """

    row_id: Hashable
    entity_id: Hashable
    as_of: object
    point: np.ndarray


@dataclass(frozen=True)
class NeighborFeatures:
    """Historical-neighbor features for one prediction row.

    The result is always aligned one-to-one with the input prediction rows,
    including rows whose eligible reference population is empty. For an empty
    population, the row is retained and the aggregate features and citations
    are set to missing values.

    Attributes:
        row_id: The :attr:`PredictionRow.row_id` of the corresponding input
            row.
        n_neighbors: The number of neighbors used to compute the features.
            ``0`` when the eligible reference population is empty.
        target_mean: The mean of the neighbors' target values, specifically the
            values that were published and not retracted at the prediction
            row's timestamp. ``None`` when there are no neighbors.
        nearest_distance: The distance to the closest neighbor under the
            configured distance function. ``None`` when there are no neighbors.
        average_distance: The mean distance across all returned neighbors.
            ``None`` when there are no neighbors.
        neighbor_ids: The identifiers of the returned neighbors, ordered from
            nearest to farthest. Empty when there are no neighbors.
        neighbor_distances: The distances of the returned neighbors, aligned
            position-by-position with :attr:`neighbor_ids`. Empty when there are
            no neighbors.

    """

    row_id: Hashable
    n_neighbors: int
    target_mean: Optional[float]
    nearest_distance: Optional[float]
    average_distance: Optional[float]
    neighbor_ids: tuple = field(default_factory=tuple)
    neighbor_distances: tuple = field(default_factory=tuple)


def _validate_and_index_target_events(
    target_events: Sequence[TargetEvent],
    record_ids: set,
) -> dict:
    """Group target events by record and validate them.

    Returns a mapping of record_id to a tuple of (sorted timestamps, values),
    where each value is a float for publish events and None for retraction
    events. Input order breaks ties between events sharing a timestamp.
    """
    grouped: dict[Hashable, list] = {}
    for event in target_events:
        if event.record_id not in record_ids:
            raise ValueError(
                f"Target event references unknown record_id {event.record_id!r}; "
                "every target event must reference a reference record."
            )
        if event.event_type is TargetEventType.PUBLISH:
            if event.value is None:
                raise ValueError(
                    f"Publish event for record_id {event.record_id!r} at timestamp "
                    f"{event.timestamp!r} is missing a value."
                )
            value: Optional[float] = float(event.value)
        elif event.event_type is TargetEventType.RETRACT:
            value = None
        else:  # pragma: no cover - defensive, Enum is closed.
            raise ValueError(f"Unknown target event type: {event.event_type!r}.")
        grouped.setdefault(event.record_id, []).append((event.timestamp, value))
    histories = {}
    for record_id, events in grouped.items():
        events.sort(key=lambda event: event[0])
        histories[record_id] = (
            [timestamp for timestamp, _ in events],
            [value for _, value in events],
        )
    return histories


def _target_value_as_of(history: tuple, as_of: object) -> tuple:
    """Resolve the target value known at as_of from a record's event history.

    Returns a tuple of (has_value, value). A record has no usable value when
    no event had been published by as_of, or the latest known event is a
    retraction.
    """
    timestamps, values = history
    # Events at as_of are already published, so use bisect_right to include
    # every event with timestamp <= as_of. Input order breaks timestamp ties,
    # which is preserved by the stable sort in _validate_and_index_target_events.
    cutoff = bisect_right(timestamps, as_of)
    if cutoff == 0:
        return (False, None)
    latest_value = values[cutoff - 1]
    if latest_value is None:
        # The latest known event was a retraction.
        return (False, None)
    return (True, latest_value)


def construct_hnsw_neighbor_features(
    prediction_rows: Sequence[PredictionRow],
    reference_records: Sequence[ReferenceRecord],
    target_events: Sequence[TargetEvent],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    k: int = 10,
    m: int = 16,
    ef_construction: int = 200,
    m0: Optional[int] = None,
    seed: Optional[int] = None,
    query_ef: Optional[int] = None,
) -> list[NeighborFeatures]:
    """Construct point-in-time historical-neighbor features for prediction rows.

    For each prediction row, an :class:`~datasketch.hnsw.HNSW` index is built
    from the reference records that satisfy all of the following at the row's
    ``as_of`` timestamp:

    * The record's ``as_of`` is strictly earlier than the row's ``as_of``.
    * The record belongs to a different entity than the row.
    * The record's target had a published value and that value had not been
      retracted. Only target events with timestamps at or before the row's
      ``as_of`` are considered, so future publications, revisions, and
      retractions cannot affect the row.

    The row's feature point is then queried against that index using the
    native HNSW approximate nearest neighbor semantics. Prediction rows are
    never inserted into a reference index, and indexes are built from scratch
    per distinct ``(as_of, entity_id)`` combination rather than incrementally.

    Args:
        prediction_rows: The rows to build features for. Results are returned
            in the same order.
        reference_records: The candidate historical reference records. Record
            IDs must be unique.
        target_events: The publication, revision, and retraction history of the
            reference records' target values, kept separate from the reference
            records themselves. Every event must reference an existing
            :attr:`ReferenceRecord.record_id`.
        distance_func: The distance function passed to
            :class:`~datasketch.hnsw.HNSW`. It is also used to compute the
            reported nearest and average distances.
        k (int): The maximum number of nearest neighbors to return per row.
        m (int): See :class:`~datasketch.hnsw.HNSW`.
        ef_construction (int): See :class:`~datasketch.hnsw.HNSW`.
        m0 (Optional[int]): See :class:`~datasketch.hnsw.HNSW`.
        seed (Optional[int]): See :class:`~datasketch.hnsw.HNSW`.
        query_ef (Optional[int]): The search-time ``ef`` passed to
            :meth:`~datasketch.hnsw.HNSW.query`. If None, the construction
            ``ef_construction`` is used, matching the native HNSW default.

    Returns:
        list[NeighborFeatures]: One :class:`NeighborFeatures` per prediction
        row, in input order. Rows with an empty eligible reference population
        are retained and carry missing values (``n_neighbors=0``,
        ``target_mean``/``nearest_distance``/``average_distance`` as ``None``,
        and empty neighbor citation tuples).

    Raises:
        ValueError: If ``k`` is not positive, a reference record ID is
            duplicated, a target event references an unknown record, or a
            publish event has no value.

    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    # Index the reference records and validate ID uniqueness.
    records_by_id: dict[Hashable, ReferenceRecord] = {}
    for record in reference_records:
        if record.record_id in records_by_id:
            raise ValueError(
                f"Duplicate reference record_id {record.record_id!r}; reference record IDs must be unique."
            )
        records_by_id[record.record_id] = record
    record_ids = set(records_by_id.keys())

    # Target publication history is kept separate from reference membership.
    target_histories = _validate_and_index_target_events(target_events, record_ids)

    # Cache one freshly built index per distinct (as_of, entity_id)
    # population, along with the as-of target values of its members.
    # A None value marks an empty population, for which no index is built.
    population_cache: dict[tuple, Optional[tuple]] = {}
    results: list[NeighborFeatures] = []

    for row in prediction_rows:
        cache_key = (row.as_of, row.entity_id)
        cached_population = population_cache.get(cache_key)
        if cached_population is None:
            if cache_key in population_cache:
                # Cached empty population.
                results.append(
                    NeighborFeatures(
                        row_id=row.row_id,
                        n_neighbors=0,
                        target_mean=None,
                        nearest_distance=None,
                        average_distance=None,
                    )
                )
                continue

            eligible: list[tuple] = []
            for record in reference_records:
                # Strictly earlier timestamps only.
                if not record.as_of < row.as_of:
                    continue
                # Different entities only.
                if record.entity_id == row.entity_id:
                    continue
                # The target must have a published, non-retracted value that
                # was known at the row's timestamp.
                history = target_histories.get(record.record_id)
                if history is None:
                    continue
                has_value, target_value = _target_value_as_of(history, row.as_of)
                if not has_value:
                    continue
                eligible.append((record, target_value))

            if not eligible:
                population_cache[cache_key] = None
                results.append(
                    NeighborFeatures(
                        row_id=row.row_id,
                        n_neighbors=0,
                        target_mean=None,
                        nearest_distance=None,
                        average_distance=None,
                    )
                )
                continue

            # Build a fresh index from the eligible reference population.
            # Prediction rows are never inserted.
            index = HNSW(
                distance_func=distance_func,
                m=m,
                ef_construction=ef_construction,
                m0=m0,
                seed=seed,
            )
            target_values: dict[Hashable, float] = {}
            for record, target_value in eligible:
                index.insert(record.record_id, record.point)
                target_values[record.record_id] = target_value
            cached_population = (index, target_values)
            population_cache[cache_key] = cached_population

        index, target_values = cached_population
        # Native HNSW approximate nearest neighbor query; k is capped at the
        # population size by HNSW.query itself.
        query_result = index.query(row.point, k=k, ef=query_ef)
        neighbor_ids = tuple(record_id for record_id, _ in query_result)
        neighbor_distances = tuple(distance for _, distance in query_result)
        n_neighbors = len(query_result)
        if n_neighbors == 0:
            # Defensive: an empty population is handled above, so an index
            # always contains at least one point.
            results.append(
                NeighborFeatures(
                    row_id=row.row_id,
                    n_neighbors=0,
                    target_mean=None,
                    nearest_distance=None,
                    average_distance=None,
                )
            )
            continue
        target_mean = float(sum(target_values[record_id] for record_id in neighbor_ids) / n_neighbors)
        nearest_distance = float(neighbor_distances[0])
        average_distance = float(sum(neighbor_distances) / n_neighbors)
        results.append(
            NeighborFeatures(
                row_id=row.row_id,
                n_neighbors=n_neighbors,
                target_mean=target_mean,
                nearest_distance=nearest_distance,
                average_distance=average_distance,
                neighbor_ids=neighbor_ids,
                neighbor_distances=neighbor_distances,
            )
        )

    return results
