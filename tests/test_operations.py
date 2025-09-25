import pytest

import copy
import tempfile

import dataclasses
import typing as tp

from functools import partial

from compgraph import operations as ops


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class JoinCase:
    joiner: ops.Joiner
    join_keys: tp.Sequence[str]
    data_left: list[ops.TRow]
    data_right: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    join_data_left_items: tuple[int, ...] = (0,)
    join_data_right_items: tuple[int, ...] = (0,)
    join_ground_truth_items: tuple[int, ...] = (0,)


OUTER_JOIN_CASES = [
    JoinCase(
        joiner=ops.OuterJoiner(),
        join_keys=("id",),
        data_left=[
            {"id": 1, "name": "Alice", "performance": 10},
            {"id": 2, "name": "Bob", "performance": 20},
            {"id": 3, "name": "Charlie", "performance": 30},
            {"id": 3, "name": "Charlie", "performance": 15},
        ],
        data_right=[
            {"id": 2, "age": 25},
            {"id": 3, "age": 30},
            {"id": 4, "age": 35},
        ],
        ground_truth=[
            {"id": 1, "name": "Alice", "performance": 10},
            {"id": 2, "name": "Bob", "age": 25, "performance": 20},
            {"id": 3, "name": "Charlie", "age": 30, "performance": 30},
            {"id": 3, "name": "Charlie", "age": 30, "performance": 15},
            {"id": 4, "age": 35},
        ],
        cmp_keys=("id",),
        join_data_left_items=(2, 3),
        join_data_right_items=(1,),
        join_ground_truth_items=(2, 3),
    ),
    JoinCase(
        joiner=ops.OuterJoiner(),
        join_keys=("id",),
        data_left=[
            {"id": 2, "age": 25},
            {"id": 3, "age": 30},
            {"id": 4, "age": 35},
        ],
        data_right=[
            {"id": 1, "name": "Alice", "performance": 10},
            {"id": 2, "name": "Bob", "performance": 20},
            {"id": 3, "name": "Charlie", "performance": 30},
            {"id": 3, "name": "Charlie", "performance": 15},
        ],
        ground_truth=[
            {"id": 1, "name": "Alice", "performance": 10},
            {"id": 2, "name": "Bob", "age": 25, "performance": 20},
            {"id": 3, "name": "Charlie", "age": 30, "performance": 30},
            {"id": 3, "name": "Charlie", "age": 30, "performance": 15},
            {"id": 4, "age": 35},
        ],
        cmp_keys=("id",),
        join_data_left_items=(1,),
        join_data_right_items=(2, 3),
        join_ground_truth_items=(2, 3),
    ),
]


@pytest.mark.parametrize("case", OUTER_JOIN_CASES)
def test_outer_join(case: JoinCase) -> None:
    joiner_data_left_rows = [
        copy.deepcopy(case.data_left[i]) for i in case.join_data_left_items
    ]
    joiner_data_right_rows = [
        copy.deepcopy(case.data_right[i]) for i in case.join_data_right_items
    ]
    joiner_ground_truth_rows = [
        copy.deepcopy(case.ground_truth[i]) for i in case.join_ground_truth_items
    ]

    key_func = _Key(*case.cmp_keys)

    joiner_result = case.joiner(
        case.join_keys, iter(joiner_data_left_rows), iter(joiner_data_right_rows)
    )
    assert isinstance(joiner_result, tp.Iterator)
    assert sorted(joiner_result, key=key_func) == sorted(
        joiner_ground_truth_rows, key=key_func
    )

    result = ops.Join(case.joiner, case.join_keys)(
        iter(case.data_left), iter(case.data_right)
    )
    assert isinstance(result, tp.Iterator)
    assert sorted(result, key=key_func) == sorted(case.ground_truth, key=key_func)


class SomeJoiner(ops.Joiner):
    def __call__(
        self,
        keys: tp.Sequence[str],
        rows_a: ops.TRowsIterable,
        rows_b: ops.TRowsIterable,
    ) -> ops.TRowsGenerator:
        raise NotImplementedError


UNKNOWN_JOIN_CASES = [
    JoinCase(
        joiner=SomeJoiner(),
        join_keys=("id",),
        data_left=[
            {"id": 1, "name": "Alice", "performance": 10},
            {"id": 2, "name": "Bob", "performance": 20},
            {"id": 3, "name": "Charlie", "performance": 30},
            {"id": 3, "name": "Charlie", "performance": 15},
        ],
        data_right=[
            {"id": 2, "age": 25},
            {"id": 3, "age": 30},
            {"id": 4, "age": 35},
        ],
        ground_truth=[
            {"id": 1, "name": "Alice", "performance": 10},
            {"id": 2, "name": "Bob", "age": 25, "performance": 20},
            {"id": 3, "name": "Charlie", "age": 30, "performance": 30},
            {"id": 3, "name": "Charlie", "age": 30, "performance": 15},
            {"id": 4, "age": 35},
        ],
        cmp_keys=("id",),
        join_data_left_items=(2, 3),
        join_data_right_items=(1,),
        join_ground_truth_items=(2, 3),
    ),
]


@pytest.mark.parametrize("case", UNKNOWN_JOIN_CASES)
def test_unknown_join(case: JoinCase) -> None:
    with pytest.raises(ValueError):
        next(
            ops.Join(case.joiner, case.join_keys)(
                iter(case.data_left), iter(case.data_right)
            )
        )


@dataclasses.dataclass
class ReadCase:
    data: str
    parser: tp.Callable[[str], ops.TRow]
    ground_truth: list[ops.TRow]


def parser(row: str, separator: str = ",") -> ops.TRow:
    data = row.split(separator)
    return {"id": int(data[0]), "name": data[1], "age": int(data[2])}


READ_CASES = [
    ReadCase(
        data="1,Alice,25\n2,Bob,30\n3,Charlie,35\n",
        parser=partial(parser, separator=","),
        ground_truth=[
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob", "age": 30},
            {"id": 3, "name": "Charlie", "age": 35},
        ],
    ),
    ReadCase(
        data="1:Alice:25\n2:Bob:30\n3:Charlie:35\n",
        parser=partial(parser, separator=":"),
        ground_truth=[
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob", "age": 30},
            {"id": 3, "name": "Charlie", "age": 35},
        ],
    ),
]


@pytest.mark.parametrize("case", READ_CASES)
def test_file_reader(case: ReadCase) -> None:
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        f.write(case.data)
        f.seek(0)
        reader = ops.Read(f.name, case.parser)
        for row, gt in zip(reader(), case.ground_truth):
            assert row == gt
