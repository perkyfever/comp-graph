import pytest
from pytest import approx

from operator import itemgetter
from itertools import islice, cycle

import dataclasses
import typing as tp

import tempfile
from functools import partial

from compgraph import graph
from compgraph import operations as ops
from compgraph.algorithms import yandex_maps_graph


@dataclasses.dataclass
class GraphReadCase:
    data: str
    parser: tp.Callable[[str], ops.TRow]
    ground_truth_a: list[ops.TRow]
    ground_truth_b: list[ops.TRow]


def parser(row: str, separator: str = ",") -> ops.TRow:
    data = row.split(separator)
    return {"doc_id": int(data[0]), "text": data[1]}


GRAPH_READ_CASES = [
    GraphReadCase(
        data="1:hello, my little WORLD\n2:Hello, my little little hell\n",
        parser=partial(parser, separator=":"),
        ground_truth_a=[
            {"count": 1, "text": "hell"},
            {"count": 1, "text": "world"},
            {"count": 2, "text": "hello"},
            {"count": 2, "text": "my"},
            {"count": 3, "text": "little"},
        ],
        ground_truth_b=[{"count": 9}],
    ),
    GraphReadCase(
        data="1/hello, my little WORLD\n2/Hello, my little little hell\n",
        parser=partial(parser, separator="/"),
        ground_truth_a=[
            {"count": 1, "text": "hell"},
            {"count": 1, "text": "world"},
            {"count": 2, "text": "hello"},
            {"count": 2, "text": "my"},
            {"count": 3, "text": "little"},
        ],
        ground_truth_b=[{"count": 9}],
    ),
]


@pytest.mark.parametrize("case", GRAPH_READ_CASES)
def test_graph_read(case: GraphReadCase) -> None:
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        text_column = "text"
        count_column = "count"
        f.write(case.data)
        f.seek(0)
        g = (
            graph.Graph.graph_from_file(f.name, case.parser)
            .map(ops.FilterPunctuation(text_column))
            .map(ops.LowerCase(text_column))
            .map(ops.Split(text_column))
            .sort([text_column])
            .reduce(ops.Count(count_column), [text_column])
            .sort([count_column, text_column])
        )
        output = g.run()
        assert list(output) == case.ground_truth_a

        another_g = g.reduce(ops.Sum(count_column), keys=[])
        output = another_g.run()
        assert list(output) == case.ground_truth_b

        output = g.run()
        assert list(output) == case.ground_truth_a


def test_yandex_maps() -> None:
    graph = yandex_maps_graph(
        "travel_time",
        "edge_length",
        enter_time_column="enter_time",
        leave_time_column="leave_time",
        edge_id_column="edge_id",
        start_coord_column="start",
        end_coord_column="end",
        weekday_result_column="weekday",
        hour_result_column="hour",
        speed_result_column="speed",
    )

    lengths = [
        {
            "start": [37.84870228730142, 55.73853974696249],
            "end": [37.8490418381989, 55.73832445777953],
            "edge_id": 8414926848168493057,
        },
        {
            "start": [37.524768467992544, 55.88785375468433],
            "end": [37.52415172755718, 55.88807155843824],
            "edge_id": 5342768494149337085,
        },
        {
            "start": [37.56963176652789, 55.846845586784184],
            "end": [37.57018438540399, 55.8469259692356],
            "edge_id": 5123042926973124604,
        },
        {
            "start": [37.41463478654623, 55.654487907886505],
            "end": [37.41442892700434, 55.654839486815035],
            "edge_id": 5726148664276615162,
        },
        {
            "start": [37.584684155881405, 55.78285809606314],
            "end": [37.58415022864938, 55.78177368734032],
            "edge_id": 451916977441439743,
        },
        {
            "start": [37.736429711803794, 55.62696328852326],
            "end": [37.736344216391444, 55.626937723718584],
            "edge_id": 7639557040160407543,
        },
        {
            "start": [37.83196756616235, 55.76662947423756],
            "end": [37.83191015012562, 55.766647034324706],
            "edge_id": 1293255682152955894,
        },
        {
            "start": [37.83196756616235, 55.76662947423756],
            "end": [37.83196756616235, 55.76662947423756],
            "edge_id": 1293255682152955000,
        },
    ]

    times = [
        {
            "leave_time": "20171020T112238.723000",
            "enter_time": "20171020T112237.427000",
            "edge_id": 8414926848168493057,
        },
        {
            "leave_time": "20171011T145553.040000",
            "enter_time": "20171011T145551.957000",
            "edge_id": 8414926848168493057,
        },
        {
            "leave_time": "20171020T090548.939000",
            "enter_time": "20171020T090547.463000",
            "edge_id": 8414926848168493057,
        },
        {
            "leave_time": "20171024T144101.879000",
            "enter_time": "20171024T144059.102000",
            "edge_id": 8414926848168493057,
        },
        {
            "leave_time": "20171022T131828.330000",
            "enter_time": "20171022T131820.842000",
            "edge_id": 5342768494149337085,
        },
        {
            "leave_time": "20171014T134826.836000",
            "enter_time": "20171014T134825.215000",
            "edge_id": 5342768494149337085,
        },
        {
            "leave_time": "20171010T060609.897000",
            "enter_time": "20171010T060608.344000",
            "edge_id": 5342768494149337085,
        },
        {
            "leave_time": "20171027T082600.201000",
            "enter_time": "20171027T082557.571000",
            "edge_id": 5342768494149337085,
        },
        {
            "leave_time": "20171027T082600",
            "enter_time": "20171027T082600",
            "edge_id": 1293255682152955000,
        },
    ]

    expected = [
        {"weekday": "Fri", "hour": 8, "speed": approx(62.2322, 0.001)},
        {"weekday": "Fri", "hour": 9, "speed": approx(78.1070, 0.001)},
        {"weekday": "Fri", "hour": 11, "speed": approx(88.9552, 0.001)},
        {"weekday": "Sat", "hour": 13, "speed": approx(100.9690, 0.001)},
        {"weekday": "Sun", "hour": 13, "speed": approx(21.8577, 0.001)},
        {"weekday": "Tue", "hour": 6, "speed": approx(105.3901, 0.001)},
        {"weekday": "Tue", "hour": 14, "speed": approx(41.5145, 0.001)},
        {"weekday": "Wed", "hour": 14, "speed": approx(106.4505, 0.001)},
    ]

    result = graph.run(
        travel_time=lambda: islice(cycle(iter(times)), len(times)),
        edge_length=lambda: iter(lengths),
    )

    assert sorted(result, key=itemgetter("weekday", "hour")) == expected

    lengths = [
        {
            "start": [37.84870228730142, 55.73853974696249],
            "end": [37.8490418381989, 55.73832445777953],
            "edge_id": 8414926848168493057,
        }
    ]

    times = [
        {
            "leave_time": "20171020T112238123.123",
            "enter_time": "20171020T112237.427000",
            "edge_id": 8414926848168493057,
        },
    ]

    result = list(
        graph.run(travel_time=lambda: iter(times), edge_length=lambda: iter(lengths))
    )
    assert result == []
