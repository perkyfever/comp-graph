import typing as tp

from . import operations as ops
from . import external_sort as ext_sort


class Graph:
    """Computational graph implementation"""

    def __init__(
        self,
        operations: list[ops.Operation] | None = None,
        join_graphs: list["Graph"] | None = None,
    ) -> None:
        self._operations = operations or []
        self._join_graphs = join_graphs or []

    @staticmethod
    def graph_from_iter(name: str) -> "Graph":
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        return Graph(operations=[ops.ReadIterFactory(name)])

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> "Graph":
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        return Graph(operations=[ops.Read(filename, parser)])

    def map(self, mapper: ops.Mapper) -> "Graph":
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        return Graph(
            operations=[*self._operations, ops.Map(mapper)],
            join_graphs=[*self._join_graphs],
        )

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> "Graph":
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        return Graph(
            operations=[*self._operations, ops.Reduce(reducer, keys)],
            join_graphs=[*self._join_graphs],
        )

    def sort(self, keys: tp.Sequence[str]) -> "Graph":
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        return Graph(
            operations=[*self._operations, ext_sort.ExternalSort(keys)],
            join_graphs=[*self._join_graphs],
        )

    def join(
        self, joiner: ops.Joiner, join_graph: "Graph", keys: tp.Sequence[str]
    ) -> "Graph":
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        return Graph(
            operations=[*self._operations, ops.Join(joiner, keys)],
            join_graphs=[*self._join_graphs, join_graph],
        )

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""

        join_graphs_cnt: int = 0
        data: ops.TRowsGenerator | None = None
        for operation in self._operations:
            if isinstance(operation, ops.Read | ops.ReadIterFactory):
                data = operation(**kwargs)
            elif isinstance(operation, ops.Join):
                assert join_graphs_cnt < len(self._join_graphs)
                assert data is not None
                join_graph = self._join_graphs[join_graphs_cnt]
                join_graphs_cnt += 1
                data = operation(data, join_graph.run(**kwargs))
            else:
                assert data is not None
                data = operation(data)

        assert data is not None

        return data
