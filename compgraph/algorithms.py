from . import Graph, operations


def word_count_graph(
    input_stream_name: str, text_column: str = "text", count_column: str = "count"
) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return (
        Graph.graph_from_iter(input_stream_name)
        .map(operations.FilterPunctuation(text_column))
        .map(operations.LowerCase(text_column))
        .map(operations.Split(text_column))
        .sort([text_column])
        .reduce(operations.Count(count_column), [text_column])
        .sort([count_column, text_column])
    )


def inverted_index_graph(
    input_stream_name: str,
    doc_column: str = "doc_id",
    text_column: str = "text",
    result_column: str = "tf_idf",
) -> Graph:
    """Constructs graph which calculates tf-idf for every word/document pair"""

    split_words = (
        Graph.graph_from_iter(input_stream_name)
        .map(operations.FilterPunctuation(text_column))
        .map(operations.LowerCase(text_column))
        .map(operations.Split(text_column))
    )

    count_docs = Graph.graph_from_iter(input_stream_name).reduce(
        operations.Count("doc_count"), keys=[]
    )

    count_idf = (
        split_words.sort([doc_column, text_column])
        .reduce(operations.FirstReducer(), keys=[doc_column, text_column])
        .sort([text_column])
        .reduce(operations.Count("doc_word_count"), keys=[text_column])
        .join(operations.InnerJoiner(), count_docs, keys=[])
        .map(operations.Division("doc_count", "doc_word_count", "inv_doc_word_freq"))
        .map(operations.Logarithm("inv_doc_word_freq", "idf"))
    )

    count_tf = (
        split_words.sort([doc_column])
        .reduce(operations.TermFrequency(text_column, "tf"), keys=[doc_column])
        .sort([text_column])
    )

    tf_idf = (
        count_idf.sort([text_column])
        .join(operations.InnerJoiner(), count_tf, keys=[text_column])
        .map(operations.Product(["tf", "idf"], result_column))
        .map(operations.Project([doc_column, text_column, result_column]))
        .sort([text_column])
        .reduce(operations.TopN(result_column, 3), keys=[text_column])
        .sort([doc_column])
        .reduce(operations.TopN(result_column, 3), keys=[doc_column])
    )

    return tf_idf


def pmi_graph(
    input_stream_name: str,
    doc_column: str = "doc_id",
    text_column: str = "text",
    result_column: str = "pmi",
) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""

    filtered_words = (
        Graph.graph_from_iter(input_stream_name)
        .map(operations.FilterPunctuation(text_column))
        .map(operations.LowerCase(text_column))
        .map(operations.Split(text_column))
        .sort([doc_column, text_column])
        .reduce(operations.Count("word_doc_cnt"), keys=[doc_column, text_column])
        .map(
            operations.Filter(
                lambda row: row["word_doc_cnt"] > 1 and len(row[text_column]) > 4
            )
        )
        .map(operations.Project([doc_column, text_column, "word_doc_cnt"]))
    )

    docs_len = (
        filtered_words.sort([doc_column])
        .reduce(operations.Sum("word_doc_cnt"), keys=[doc_column])
        .map(operations.Rename("word_doc_cnt", "doc_len"))
    )

    words_doc_freq = (
        filtered_words.sort([doc_column])
        .join(operations.InnerJoiner(), docs_len, keys=[doc_column])
        .map(operations.Division("word_doc_cnt", "doc_len", "word_doc_freq"))
    )

    doc_total_len = docs_len.reduce(operations.Sum("doc_len"), keys=[]).map(
        operations.Rename("doc_len", "doc_total_len")
    )

    words_total_freq = (
        filtered_words.sort([text_column])
        .reduce(operations.Sum("word_doc_cnt"), keys=[text_column])
        .map(operations.Rename("word_doc_cnt", "word_total_cnt"))
        .join(operations.InnerJoiner(), doc_total_len, keys=[])
        .map(operations.Division("word_total_cnt", "doc_total_len", "word_total_freq"))
        .map(operations.Project([text_column, "word_total_freq"]))
    )

    words_pmi = (
        words_doc_freq.sort([text_column])
        .join(operations.InnerJoiner(), words_total_freq, keys=[text_column])
        .map(
            operations.Division(
                "word_doc_freq", "word_total_freq", "word_freq_quotient"
            )
        )
        .map(operations.Logarithm("word_freq_quotient", "pmi"))
        .sort([doc_column])
        .reduce(operations.TopN("pmi", 10), keys=[doc_column])
        .map(operations.Project([doc_column, text_column, "pmi"]))
        .sort([doc_column])
    )

    return words_pmi


def yandex_maps_graph(
    input_stream_name_time: str,
    input_stream_name_length: str,
    enter_time_column: str = "enter_time",
    leave_time_column: str = "leave_time",
    edge_id_column: str = "edge_id",
    start_coord_column: str = "start",
    end_coord_column: str = "end",
    weekday_result_column: str = "weekday",
    hour_result_column: str = "hour",
    speed_result_column: str = "speed",
) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""

    edge_with_dist = (
        Graph.graph_from_iter(input_stream_name_length)
        .map(operations.Haversine(start_coord_column, end_coord_column, "edge_length"))
        .map(operations.Project([edge_id_column, "edge_length"]))
        .sort([edge_id_column])
    )

    logs_with_time = (
        Graph.graph_from_iter(input_stream_name_time)
        .map(operations.Hour(enter_time_column, hour_result_column))
        .map(operations.Weekday(enter_time_column, weekday_result_column))
        .map(
            operations.Filter(
                lambda row: row.get(hour_result_column, None) is not None
                and row.get(weekday_result_column, None) is not None
            )
        )
        .map(
            operations.TimeDifference(
                enter_time_column, leave_time_column, "travel_time"
            )
        )
        .map(operations.Filter(lambda row: row.get("travel_time", -1) >= 0))
    )

    logs_with_total_time = (
        logs_with_time.sort([hour_result_column, weekday_result_column])
        .reduce(
            operations.Sum("travel_time"),
            keys=[hour_result_column, weekday_result_column],
        )
        .map(operations.Rename("travel_time", "total_time"))
        .map(
            operations.Project(
                [hour_result_column, weekday_result_column, "total_time"]
            )
        )
    )

    logs_with_total_dist = (
        logs_with_time.sort([edge_id_column])
        .join(operations.InnerJoiner(), edge_with_dist, keys=[edge_id_column])
        .sort([hour_result_column, weekday_result_column])
        .reduce(
            operations.Sum("edge_length"),
            keys=[hour_result_column, weekday_result_column],
        )
        .map(operations.Rename("edge_length", "total_dist"))
        .map(
            operations.Project(
                [hour_result_column, weekday_result_column, "total_dist"]
            )
        )
    )

    average_speed = (
        logs_with_total_time.join(
            operations.InnerJoiner(),
            logs_with_total_dist,
            keys=[hour_result_column, weekday_result_column],
        )
        .map(operations.Division("total_dist", "total_time", speed_result_column))
        .map(operations.ToCalendarWeekday(weekday_result_column))
        .map(
            operations.Project(
                [hour_result_column, weekday_result_column, speed_result_column]
            )
        )
        .map(operations.Normalize(speed_result_column, 3.6))
        .sort([hour_result_column, weekday_result_column])
    )

    return average_speed
