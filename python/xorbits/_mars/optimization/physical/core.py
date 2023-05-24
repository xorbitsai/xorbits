# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import functools
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Set, Tuple, Type, Union

from ...core import ChunkGraph, ChunkType, OperandType
from ...utils import build_fuse_chunk

REDUCTION = object()


@dataclasses.dataclass
class _Fuse:
    graph: ChunkGraph
    heads: List[ChunkType]
    tails: List[ChunkType]


class RuntimeOptimizer(ABC):
    engine = None

    def __init__(self, graph: ChunkGraph):
        self._graph = graph

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check this optimizer is available.

        Returns
        -------
        is_available : bool
            Available.
        """

    @abstractmethod
    def optimize(self) -> Tuple[List[_Fuse], List[ChunkType]]:
        """
        Optimize chunk graph.
        """

    def _fuse_nodes(
        self, fuses: List[List[ChunkType]], fuse_cls: OperandType
    ) -> Tuple[List[List[ChunkType]], List[ChunkType]]:
        graph = self._graph
        fused_nodes = []

        for fuse in fuses:
            head_node = fuse[0]
            tail_node = fuse[-1]

            fused_chunk = build_fuse_chunk(
                fuse, fuse_cls, op_kw={"dtype": tail_node.dtype}
            ).data
            graph.add_node(fused_chunk)
            for node in graph.iter_successors(tail_node):
                graph.add_edge(fused_chunk, node)
            for node in graph.iter_predecessors(head_node):
                graph.add_edge(node, fused_chunk)
            for node in fuse:
                graph.remove_node(node)
            fused_nodes.append(fused_chunk)

            try:
                # check tail node if it's in results
                i = graph.results.index(tail_node)
                graph.results[i] = fused_chunk
            except ValueError:
                pass

        return fuses, fused_nodes


class GraphTraversalOptimizer(RuntimeOptimizer):
    @abstractmethod
    def _can_fuse(self, node: ChunkType) -> Union[bool, object]:
        """
        Check if a node can be fused

        Returns
        -------
        bool or object:
            If the node can be fused and is a reduction operation, return
            REDUCTION. Otherwise, if the node can be fused and is not a
            reduction operation, return True. For other nodes, return False
        """

    def _collect_fuse(
        self,
        graph: ChunkGraph,
        node: ChunkType,
        graph_results: Set[ChunkType],
        cached_can_fuse: Callable,
    ) -> _Fuse:
        fuse_graph = ChunkGraph()
        fuse_graph.add_node(node)
        fuse_heads: ChunkType = []
        fuse_tails: ChunkType = []
        tail_reduction_node: ChunkType = None

        stack: List[ChunkType] = [node]
        # Do a full search of sub graph even the fuse tails > 1
        while len(stack) != 0:
            node = stack.pop()
            is_head = graph.count_predecessors(node) == 0
            for n in graph.iter_predecessors(node):
                can_fuse = cached_can_fuse(n)
                if can_fuse is False or can_fuse is REDUCTION:
                    is_head = True
                elif not fuse_graph.contains(n):
                    stack.append(n)
                    fuse_graph.add_node(n)
                else:
                    fuse_graph.add_edge(n, node)
            if is_head:
                fuse_heads.append(node)
            # Skip the successors of tail reduction node.
            if node is tail_reduction_node:
                continue
            is_tail = graph.count_successors(node) == 0 or node in graph_results
            for n in graph.iter_successors(node):
                can_fuse = cached_can_fuse(n)
                if can_fuse is False:
                    is_tail = True
                elif can_fuse is REDUCTION:
                    if tail_reduction_node is None:
                        tail_reduction_node = n
                        fuse_tails.append(n)
                        stack.append(n)
                        fuse_graph.add_node(n)
                    elif n is tail_reduction_node:
                        fuse_graph.add_edge(node, n)
                    else:
                        is_tail = True
                elif not fuse_graph.contains(n):
                    stack.append(n)
                    fuse_graph.add_node(n)
                else:
                    fuse_graph.add_edge(node, n)
            if is_tail:
                fuse_tails.append(node)

        return _Fuse(fuse_graph, fuse_heads, fuse_tails)

    def _graph_traverse(self, logger, logger_str: str) -> List[_Fuse]:
        fuses: List[_Fuse] = []
        explored: Set[ChunkType] = set()
        cached_can_fuse = functools.lru_cache(maxsize=None)(self._can_fuse)

        graph = self._graph
        graph_results = set(graph.results)
        for node in graph.topological_iter():
            if node.op.gpu or node.op.sparse:
                # break
                return [], []
            if node in explored or node in graph_results:
                continue
            can_fuse = cached_can_fuse(node)
            if can_fuse is True:
                fuse = self._collect_fuse(graph, node, graph_results, cached_can_fuse)
                if len(fuse.graph) > 1:
                    explored.update(fuse.graph)
                    if len(fuse.tails) == 1:
                        fuses.append(fuse)
                    else:
                        logger.info(logger_str)
        return fuses

    def _fuse_nodes(
        self, fuses: List[_Fuse], fuse_cls
    ) -> Tuple[List[_Fuse], List[ChunkType]]:
        graph = self._graph
        fused_nodes: List[ChunkType] = []

        for fuse in fuses:
            fuse_graph = fuse.graph
            tail_nodes = fuse.tails
            head_nodes = fuse.heads
            inputs = [
                inp for n in head_nodes for inp in n.inputs if inp not in fuse_graph
            ]

            tail_chunk = tail_nodes[0]
            tail_chunk_op = tail_chunk.op
            fuse_op = fuse_cls(
                sparse=tail_chunk_op.sparse,
                gpu=tail_chunk_op.gpu,
                _key=tail_chunk_op.key,
                fuse_graph=fuse_graph,
                dtype=tail_chunk.dtype,
            )
            fused_chunk = fuse_op.new_chunk(
                inputs,
                kws=[tail_chunk.params],
                _key=tail_chunk.key,
                _chunk=tail_chunk,
            ).data

            graph.add_node(fused_chunk)
            for node in graph.iter_successors(tail_chunk):
                graph.add_edge(fused_chunk, node)
            for head_chunk in head_nodes:
                for node in graph.iter_predecessors(head_chunk):
                    if not fuse_graph.contains(node):
                        graph.add_edge(node, fused_chunk)
            for node in fuse_graph:
                graph.remove_node(node)
            fused_nodes.append(fused_chunk)

            try:
                # check tail node if it's in results
                i = graph.results.index(tail_chunk)
                graph.results[i] = fused_chunk
            except ValueError:
                pass

        return fuses, fused_nodes


_engine_to_optimizers: Dict[str, Type[RuntimeOptimizer]] = dict()


def register_optimizer(optimizer_cls: Type[RuntimeOptimizer]):
    _engine_to_optimizers[optimizer_cls.engine] = optimizer_cls
    return optimizer_cls


def optimize(graph: ChunkGraph, engines: List[str] = None) -> ChunkGraph:
    if engines is None:
        engines = ["jax", "numexpr", "cupy"]

    for engine in engines:
        optimizer_cls = _engine_to_optimizers[engine]
        optimizer = optimizer_cls(graph)
        if not optimizer.is_available():
            continue
        optimizer.optimize()

    return graph
