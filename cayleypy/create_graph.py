from .graphs_lib import prepare_graph
from .cayley_graph import CayleyGraph


def create_graph(**kwargs) -> CayleyGraph:
    """Creates CayleyGraph from kwargs.

    Pass the following to kwargs:
        * "name" - the name of the graph, mandatory, see ``prepare_graph`` source for supported names,
        * other parameters of the graph (such as "n") that will be passed to ``prepare_graph``,
        * "central_state" - central state, if you need to override it,
        * any arguments that are accepted by ``CayleyGraph`` constructor (e.g. ``verbose=2``).

    All passed kwargs will be first passed to ``prepare_graph`` to construct ``CayleyGraphDef`` and then to
    ``CayleyGraph`` constructor.

    This function allows to create graphs in a uniform way. It is useful when you want to specify graph type and
    parameters in a config and have the same code handling different configs.

    This is not recommended in most cases. Instead, create ``CayleyGraphDef`` using one of library classes and then pass
    it to ``CayleyGraph`` constructor.
    """
    graph_def = prepare_graph(**kwargs)
    if "central_state" in kwargs:
        graph_def = graph_def.with_central_state(kwargs["central_state"])
    return CayleyGraph(graph_def, **kwargs)
