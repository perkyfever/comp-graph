import os
import argparse

import json

from compgraph.operations import Read
from compgraph.algorithms import yandex_maps_graph

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = os.path.join(os.path.dirname(CURRENT_PATH), "resources")
DEFAULT_TIMES_FILE = os.path.join("extract_me", "travel_times")
DEFAULT_LENGTHS_FILE = os.path.join("extract_me", "road_graph_data")


def main() -> None:

    parser = argparse.ArgumentParser(description="Run pmi graph on input file")

    parser.add_argument(
        "--times",
        default=str(os.path.join(PATH_TO_DATA, f"{DEFAULT_TIMES_FILE}.txt")),
        type=str,
        help="Input times file path",
    )

    parser.add_argument(
        "--lengths",
        default=str(os.path.join(PATH_TO_DATA, f"{DEFAULT_LENGTHS_FILE}.txt")),
        type=str,
        help="Input lengths file path",
    )

    parser.add_argument(
        "--output",
        default=str(
            os.path.join(
                PATH_TO_DATA, os.path.join("extract_me", "yandex_maps_results.txt")
            )
        ),
        type=str,
        help="Output file path",
    )

    args = parser.parse_args()
    graph = yandex_maps_graph(
        input_stream_name_time="times",
        input_stream_name_length="lengths",
        enter_time_column="enter_time",
        leave_time_column="leave_time",
        edge_id_column="edge_id",
        start_coord_column="start",
        end_coord_column="end",
        weekday_result_column="weekday",
        hour_result_column="hour",
        speed_result_column="speed",
    )

    times_filepath = args.times
    lengths_filepath = args.lengths
    output_filepath = args.output

    times_reader = Read(filename=times_filepath, parser=lambda line: json.loads(line))
    lengths_reader = Read(
        filename=lengths_filepath, parser=lambda line: json.loads(line)
    )

    result = graph.run(lengths=lambda: lengths_reader(), times=lambda: times_reader())
    with open(output_filepath, "w") as out:
        for row in result:
            print(json.dumps(row), file=out)


if __name__ == "__main__":
    main()
