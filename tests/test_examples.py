import os
import json

import pytest
from pytest import approx

from pathlib import PosixPath

import subprocess

from operator import itemgetter

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_SCRIPTS = os.path.join(os.path.dirname(CURRENT_PATH), "examples")


@pytest.fixture
def word_count_file_name(tmp_path: PosixPath) -> str:
    rows = [
        {"doc_id": 1, "text": "hello, my little WORLD"},
        {"doc_id": 2, "text": "Hello, my little little hell"},
    ]
    data = "\n".join(json.dumps(row) for row in rows)
    input_file = tmp_path / "input.txt"
    input_file.write_text(data)
    return str(input_file)


def test_word_count_graph(word_count_file_name: str, tmp_path: PosixPath) -> None:
    expected = [
        {"count": 1, "text": "hell"},
        {"count": 1, "text": "world"},
        {"count": 2, "text": "hello"},
        {"count": 2, "text": "my"},
        {"count": 3, "text": "little"},
    ]
    output_file = tmp_path / "output.txt"
    script = subprocess.run(
        [
            "python",
            os.path.join(PATH_TO_SCRIPTS, "run_word_count.py"),
            "--input",
            word_count_file_name,
            "--output",
            output_file,
        ],
        capture_output=True,
    )

    assert script.returncode == 0

    with open(output_file) as out:
        output = out.read()
        result = [json.loads(row) for row in output.splitlines()]
        assert result == expected


@pytest.fixture
def inverted_index_file_name(tmp_path: PosixPath) -> str:
    rows = [
        {"doc_id": 1, "text": "hello, little world"},
        {"doc_id": 2, "text": "little"},
        {"doc_id": 3, "text": "little little little"},
        {"doc_id": 4, "text": "little? hello little world"},
        {"doc_id": 5, "text": "HELLO HELLO! WORLD..."},
        {"doc_id": 6, "text": "world? world... world!!! WORLD!!! HELLO!!!"},
    ]
    data = "\n".join(json.dumps(row) for row in rows)
    input_file = tmp_path / "input.txt"
    input_file.write_text(data)
    return str(input_file)


def test_inverted_index_graph(
    inverted_index_file_name: str, tmp_path: PosixPath
) -> None:
    expected = [
        {"doc_id": 1, "text": "hello", "tf_idf": approx(0.1351, 0.001)},
        {"doc_id": 1, "text": "world", "tf_idf": approx(0.1351, 0.001)},
        {"doc_id": 2, "text": "little", "tf_idf": approx(0.4054, 0.001)},
        {"doc_id": 3, "text": "little", "tf_idf": approx(0.4054, 0.001)},
        {"doc_id": 4, "text": "hello", "tf_idf": approx(0.1013, 0.001)},
        {"doc_id": 4, "text": "little", "tf_idf": approx(0.2027, 0.001)},
        {"doc_id": 5, "text": "hello", "tf_idf": approx(0.2703, 0.001)},
        {"doc_id": 5, "text": "world", "tf_idf": approx(0.1351, 0.001)},
        {"doc_id": 6, "text": "world", "tf_idf": approx(0.3243, 0.001)},
    ]
    output_file = tmp_path / "output.txt"
    script = subprocess.run(
        [
            "python",
            os.path.join(PATH_TO_SCRIPTS, "run_inverted_index.py"),
            "--input",
            inverted_index_file_name,
            "--output",
            output_file,
        ],
        capture_output=True,
    )

    assert script.returncode == 0

    with open(output_file) as out:
        output = out.read()
        result = [json.loads(row) for row in output.splitlines()]
        assert sorted(result, key=itemgetter("doc_id", "text")) == expected


@pytest.fixture
def pmi_file_name(tmp_path: PosixPath) -> str:
    rows = [
        {"doc_id": 1, "text": "hello, little world"},
        {"doc_id": 2, "text": "little"},
        {"doc_id": 3, "text": "little little little"},
        {"doc_id": 4, "text": "little? hello little world"},
        {"doc_id": 5, "text": "HELLO HELLO! WORLD..."},
        {
            "doc_id": 6,
            "text": "world? world... world!!! WORLD!!! HELLO!!! HELLO!!!!!!!",
        },
    ]
    data = "\n".join(json.dumps(row) for row in rows)
    input_file = tmp_path / "input.txt"
    input_file.write_text(data)
    return str(input_file)


def test_pmi_graph(pmi_file_name: str, tmp_path: PosixPath) -> None:

    expected = [
        {"doc_id": 3, "text": "little", "pmi": approx(0.9555, 0.001)},
        {"doc_id": 4, "text": "little", "pmi": approx(0.9555, 0.001)},
        {"doc_id": 5, "text": "hello", "pmi": approx(1.1786, 0.001)},
        {"doc_id": 6, "text": "world", "pmi": approx(0.7731, 0.001)},
        {"doc_id": 6, "text": "hello", "pmi": approx(0.0800, 0.001)},
    ]
    output_file = tmp_path / "output.txt"
    script = subprocess.run(
        [
            "python",
            os.path.join(PATH_TO_SCRIPTS, "run_pmi.py"),
            "--input",
            pmi_file_name,
            "--output",
            output_file,
        ],
        capture_output=True,
    )

    assert script.returncode == 0

    with open(output_file) as out:
        output = out.read()
        result = [json.loads(row) for row in output.splitlines()]
        assert result == expected


@pytest.fixture
def yandex_maps_file_names(tmp_path: PosixPath) -> tuple[str, str]:
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
    ]

    times_data = "\n".join(json.dumps(row) for row in times)
    lengths_data = "\n".join(json.dumps(row) for row in lengths)

    times_input = tmp_path / "times_input.txt"
    times_input.write_text(times_data)

    lengths_input = tmp_path / "lengths_input.txt"
    lengths_input.write_text(lengths_data)

    return str(times_input), str(lengths_input)


def test_yandex_maps_graph(
    yandex_maps_file_names: tuple[str, str], tmp_path: PosixPath
) -> None:

    times_fname, lengths_fname = yandex_maps_file_names

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
    output_file = tmp_path / "output.txt"
    script = subprocess.run(
        [
            "python",
            os.path.join(PATH_TO_SCRIPTS, "run_yandex_maps.py"),
            "--times",
            times_fname,
            "--lengths",
            lengths_fname,
            "--output",
            output_file,
        ],
        capture_output=True,
    )

    assert script.returncode == 0

    with open(output_file) as out:
        output = out.read()
        result = [json.loads(row) for row in output.splitlines()]
        assert sorted(result, key=itemgetter("weekday", "hour")) == expected
