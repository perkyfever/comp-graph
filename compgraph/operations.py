from abc import abstractmethod, ABC
import typing as tp

import re
import string

import calendar
from datetime import datetime

import heapq
import itertools as it
from collections import Counter

import numpy as np

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(
        self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any
    ) -> TRowsGenerator:
        raise NotImplementedError


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        raise NotImplementedError


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(
        self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any
    ) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper(row)


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        raise NotImplementedError


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(
        self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any
    ) -> TRowsGenerator:
        for _, group in it.groupby(
            rows, key=lambda row: tuple(row[key] for key in self.keys)
        ):
            yield from self.reducer(tuple(self.keys), group)


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = "_1", suffix_b: str = "_2") -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        raise NotImplementedError

    def _merge_rows(self, keys: tp.Sequence[str], row_a: TRow, row_b: TRow) -> TRow:
        merged_row = TRow()
        cols_a = set(row_a.keys())
        cols_b = set(row_b.keys())
        common_cols = (cols_a & cols_b) - set(keys)
        for col in keys:
            merged_row[col] = row_a[col]
        for col in common_cols:
            merged_row[col + self._a_suffix] = row_a[col]
            merged_row[col + self._b_suffix] = row_b[col]
        for col in cols_a - common_cols:
            merged_row[col] = row_a[col]
        for col in cols_b - common_cols:
            merged_row[col] = row_b[col]
        return merged_row


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    @staticmethod
    def _is_less(a: tuple[tp.Any, ...], b: tuple[tp.Any, ...]) -> bool:
        assert len(a) == len(b)
        for a_val, b_val in zip(a, b):
            if str(a_val) < str(b_val):
                return True
            if str(a_val) > str(b_val):
                return False
        return False

    def __call__(
        self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any
    ) -> TRowsGenerator:
        groups_a = it.groupby(
            rows, key=lambda row: tuple(row[key] for key in self.keys)
        )
        groups_b = it.groupby(
            args[0], key=lambda row: tuple(row[key] for key in self.keys)
        )
        group_key_b, group_b = next(groups_b, (None, None))
        if isinstance(self.joiner, InnerJoiner):
            for group_key_a, group_a in groups_a:
                while (
                    group_key_b is not None
                    and group_b is not None
                    and self._is_less(group_key_b, group_key_a)
                ):
                    group_key_b, group_b = next(groups_b, (None, None))
                if (
                    group_key_b is not None
                    and group_b is not None
                    and group_key_a == group_key_b
                ):
                    yield from self.joiner(self.keys, group_a, group_b)
                    group_key_b, group_b = next(groups_b, (None, None))
        elif isinstance(self.joiner, OuterJoiner):
            for group_key_a, group_a in groups_a:
                while (
                    group_key_b is not None
                    and group_b is not None
                    and self._is_less(group_key_b, group_key_a)
                ):
                    yield from self.joiner(self.keys, [], group_b)
                    group_key_b, group_b = next(groups_b, (None, None))
                if (
                    group_key_b is not None
                    and group_b is not None
                    and group_key_a == group_key_b
                ):
                    yield from self.joiner(self.keys, group_a, group_b)
                    group_key_b, group_b = next(groups_b, (None, None))
                else:
                    yield from self.joiner(self.keys, group_a, [])
            while group_key_b is not None and group_b is not None:
                yield from self.joiner(self.keys, [], group_b)
                group_key_b, group_b = next(groups_b, (None, None))
        elif isinstance(self.joiner, LeftJoiner):
            for group_key_a, group_a in groups_a:
                while (
                    group_key_b is not None
                    and group_b is not None
                    and self._is_less(group_key_b, group_key_a)
                ):
                    group_key_b, group_b = next(groups_b, (None, None))
                if (
                    group_key_b is not None
                    and group_b is not None
                    and group_key_a == group_key_b
                ):
                    yield from self.joiner(self.keys, group_a, group_b)
                else:
                    yield from self.joiner(self.keys, group_a, [])
        elif isinstance(self.joiner, RightJoiner):
            for group_key_a, group_a in groups_a:
                while (
                    group_key_b is not None
                    and group_b is not None
                    and self._is_less(group_key_b, group_key_a)
                ):
                    yield from self.joiner(self.keys, [], group_b)
                    group_key_b, group_b = next(groups_b, (None, None))
                if (
                    group_key_b is not None
                    and group_b is not None
                    and group_key_a == group_key_b
                ):
                    yield from self.joiner(self.keys, group_a, group_b)
            while group_key_b is not None and group_b is not None:
                yield from self.joiner(self.keys, [], group_b)
                group_key_b, group_b = next(groups_b, (None, None))
        else:
            raise ValueError("Unknown joiner type")


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class Division(Mapper):
    """Calculates quotient of two columns"""

    def __init__(
        self, column_num: str, column_den: str, result_column: str = "quotient"
    ) -> None:
        """
        :param column: name of column to process
        :param result_column: name of column to save result in
        """
        self.column_num = column_num
        self.column_den = column_den
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.column_num in mapped_row and self.column_den in mapped_row:
            mapped_row[self.result_column] = (
                mapped_row[self.column_num] / mapped_row[self.column_den]
            )
        yield mapped_row


class Logarithm(Mapper):
    """Calculates logarithm of column value"""

    def __init__(self, column: str, result_column: str = "logarithm") -> None:
        """
        :param column: name of column to process
        :param result_column: name of column to save result in
        """
        self.column = column
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.column in mapped_row:
            mapped_row[self.result_column] = float(np.log(mapped_row[self.column]))
        yield mapped_row


class FilterPunctuation(Mapper):
    """Leave only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        punctuation_filter = re.compile(f"[{re.escape(string.punctuation)}]")
        if self.column in mapped_row:
            mapped_row[self.column] = punctuation_filter.sub(
                "", str(mapped_row[self.column])
            )
        yield mapped_row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.column in mapped_row:
            mapped_row[self.column] = self._lower_case(mapped_row[self.column])
        yield mapped_row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.column in row:
            separator_pattern = r"\s+" if self.separator is None else self.separator
            matches = re.finditer(separator_pattern, row[self.column])

            start = 0
            for match in matches:
                end = match.start()
                mapped_row = row.copy()
                mapped_row[self.column] = row[self.column][start:end]
                start = match.end()
                yield mapped_row

            if start < len(row[self.column]):
                mapped_row = row.copy()
                mapped_row[self.column] = row[self.column][start:]
                yield mapped_row


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(
        self, columns: tp.Sequence[str], result_column: str = "product"
    ) -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        product = 1
        for column in self.columns:
            product *= mapped_row[column]
        mapped_row[self.result_column] = product
        yield mapped_row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = TRow()
        for column in self.columns:
            if column in row:
                mapped_row[column] = row[column]
        yield mapped_row


class Rename(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, column: str, new_column: str) -> None:
        """
        :param column: column name to rename
        :param new_column: new column name
        """
        self.column = column
        self.new_column = new_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.column in mapped_row:
            del mapped_row[self.column]
            mapped_row[self.new_column] = row[self.column]
        yield mapped_row


class Haversine(Mapper):
    """Calculates distance between two points on the Earth"""

    EARTH_RADIUS = 6373

    def __init__(
        self, a_column: str, b_column: str, result_column: str = "haversine"
    ) -> None:
        """
        :param a_column: name of column with first point
        :param b_column: name of column with second point
        :param result_column: name of column to save result in
        """
        self.a_column = a_column
        self.b_column = b_column
        self.result_column = result_column

    def _haversine(
        self, a_lat: float, a_lon: float, b_lat: float, b_lon: float
    ) -> float:
        """
        Calculate the great circle distance between two points on the Earth in meters
        :param a_lat: latitude of the first point
        :param a_lon: longitude of the first point
        :param b_lat: latitude of the second point
        :param b_lon: longitude of the second point
        """
        a_lat_rad = np.radians(a_lat)
        a_lon_rad = np.radians(a_lon)
        b_lat_rad = np.radians(b_lat)
        b_lon_rad = np.radians(b_lon)

        delta_lat = b_lat_rad - a_lat_rad
        delta_lon = b_lon_rad - a_lon_rad
        numerator = (
            1.0
            - np.cos(delta_lat)
            + np.cos(a_lat_rad) * np.cos(b_lat_rad) * (1.0 - np.cos(delta_lon))
        )

        return 2000.0 * self.EARTH_RADIUS * np.arcsin(np.sqrt(numerator / 2.0))

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.a_column in mapped_row and self.b_column in mapped_row:
            a_lon, a_lat = mapped_row[self.a_column]
            b_lon, b_lat = mapped_row[self.b_column]
            mapped_row[self.result_column] = self._haversine(a_lat, a_lon, b_lat, b_lon)
        yield mapped_row


class Hour(Mapper):
    """Get hour from timestamp"""

    def __init__(self, column: str, result_column: str) -> None:
        """
        :param column: column name to get timestamp from
        :param result_column: new column name
        """
        self.column = column
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.column in mapped_row:
            try:
                time = datetime.strptime(mapped_row[self.column], "%Y%m%dT%H%M%S.%f")
                mapped_row[self.result_column] = time.hour
            except Exception:
                try:
                    time = datetime.strptime(mapped_row[self.column], "%Y%m%dT%H%M%S")
                    mapped_row[self.result_column] = time.hour
                except Exception:
                    pass
        yield mapped_row


class Weekday(Mapper):
    """Get weekday from timestamp"""

    def __init__(self, column: str, result_column: str) -> None:
        """
        :param column: column name to get timestamp from
        :param result_column: new column name
        """
        self.column = column
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.column in mapped_row:
            try:
                time = datetime.strptime(mapped_row[self.column], "%Y%m%dT%H%M%S.%f")
                mapped_row[self.result_column] = time.weekday()
            except Exception:
                try:
                    time = datetime.strptime(mapped_row[self.column], "%Y%m%dT%H%M%S")
                    mapped_row[self.result_column] = time.weekday()
                except Exception:
                    pass
        yield mapped_row


class ToCalendarWeekday(Mapper):
    """Get calendar weekday from timestamp"""

    def __init__(self, column: str) -> None:
        """
        :param column: column name to get weekday from
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.column in mapped_row:
            mapped_row[self.column] = calendar.day_abbr[mapped_row[self.column]]
        yield mapped_row


class TimeDifference(Mapper):
    """Get time difference between two timestamps in seconds"""

    def __init__(self, start_column: str, end_column: str, result_column: str) -> None:
        """
        :param start_column: column name to get start timestamp from
        :param end_column: column name to get end timestamp from
        :param result_column: new column name
        """
        self.start_column = start_column
        self.end_column = end_column
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.start_column in mapped_row and self.end_column in mapped_row:
            try:
                start_time = datetime.strptime(
                    mapped_row[self.start_column], "%Y%m%dT%H%M%S.%f"
                )
                end_time = datetime.strptime(
                    mapped_row[self.end_column], "%Y%m%dT%H%M%S.%f"
                )
                mapped_row[self.result_column] = (end_time - start_time).total_seconds()
            except Exception:
                try:
                    start_time = datetime.strptime(
                        mapped_row[self.start_column], "%Y%m%dT%H%M%S"
                    )
                    end_time = datetime.strptime(
                        mapped_row[self.end_column], "%Y%m%dT%H%M%S"
                    )
                    mapped_row[self.result_column] = (
                        end_time - start_time
                    ).total_seconds()
                except Exception:
                    pass

        yield mapped_row


class Normalize(Mapper):
    """Normalize column values by a constant"""

    def __init__(self, column: str, coef: float) -> None:
        """
        :param column: column name to normalize
        :param coef: normalization coefficient
        """
        self.column = column
        self.coef = coef

    def __call__(self, row: TRow) -> TRowsGenerator:
        mapped_row = row.copy()
        if self.column in mapped_row:
            mapped_row[self.column] *= self.coef
        yield mapped_row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        top_n = heapq.nlargest(self.n, rows, key=lambda row: row[self.column_max])
        yield from top_n


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = "tf") -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        for row in rows:
            key_dict = {key: row[key] for key in group_key}
            counts = Counter(row[self.words_column] for row in rows)
            total_words = 1 + sum(counts.values())
            word_seen = False
            for word, count in counts.items():
                reduced_row = key_dict.copy()
                reduced_row[self.words_column] = word
                reduced_row[self.result_column] = count / total_words
                if word == row[self.words_column]:
                    word_seen = True
                    reduced_row[self.result_column] += 1 / total_words
                yield reduced_row
            if not word_seen:
                reduced_row = key_dict.copy()
                reduced_row[self.words_column] = row[self.words_column]
                reduced_row[self.result_column] = 1 / total_words
                yield reduced_row
            break


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        for row in rows:
            reduced_row = {key: row[key] for key in group_key}
            reduced_row[self.column] = 1 + sum(1 for _ in rows)
            yield reduced_row


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        for row in rows:
            reduced_row = {key: row[key] for key in group_key}
            reduced_row[self.column] = row[self.column] + sum(
                row[self.column] for row in rows
            )
            yield reduced_row
            break


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        rows_b_list = list(rows_b)
        for row_a in rows_a:
            for row_b in rows_b_list:
                yield self._merge_rows(keys, row_a, row_b)


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        row_b_list = list(rows_b)
        if len(row_b_list) == 0:
            yield from rows_a
        else:
            is_null = True
            for row_a in rows_a:
                is_null = False
                for row_b in row_b_list:
                    yield self._merge_rows(keys, row_a, row_b)
            if is_null:
                yield from row_b_list


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        row_b_list = list(rows_b)
        if len(row_b_list) == 0:
            yield from rows_a
        else:
            for row_a in rows_a:
                for row_b in row_b_list:
                    yield self._merge_rows(keys, row_a, row_b)


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        row_a_list = list(rows_a)
        if len(row_a_list) == 0:
            yield from rows_b
        else:
            for row_b in rows_b:
                for row_a in row_a_list:
                    yield self._merge_rows(keys, row_a, row_b)
