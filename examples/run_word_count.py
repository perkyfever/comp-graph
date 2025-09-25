import os
import argparse

import json

from compgraph.operations import Read
from compgraph.algorithms import word_count_graph

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = os.path.join(os.path.dirname(CURRENT_PATH), "resources")
DEFAULT_FILE = os.path.join("extract_me", "text_corpus")


def main() -> None:

    parser = argparse.ArgumentParser(description="Run word count graph input file")

    parser.add_argument(
        "--input",
        default=str(os.path.join(PATH_TO_DATA, f"{DEFAULT_FILE}.txt")),
        type=str,
        help="Input file path",
    )

    parser.add_argument(
        "--output",
        default=str(
            os.path.join(PATH_TO_DATA, f"{DEFAULT_FILE}_word_count_result.txt")
        ),
        type=str,
        help="Output file path",
    )

    args = parser.parse_args()
    graph = word_count_graph(
        input_stream_name="input", text_column="text", count_column="count"
    )

    input_filepath = args.input
    output_filepath = args.output

    file_reader = Read(filename=input_filepath, parser=lambda line: json.loads(line))

    result = graph.run(input=lambda: file_reader())
    with open(output_filepath, "w") as out:
        for row in result:
            print(json.dumps(row), file=out)


if __name__ == "__main__":
    main()
