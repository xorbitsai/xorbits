# Copyright 2022-2023 XProbe Inc.
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
import operator

from ....core import ChunkGraph
from ....tensor.arithmetic import TensorTreeAdd
from ....tensor.indexing import TensorSlice
from ....tensor.reduction import TensorSum
from ..jax import JAXRuntimeOptimizer


def test_jax():
    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        @                   @                          @
          \               /                          /
            @ --> @ --> S      ========>     # --> S
          /               \                          \
        @                   @                          @

        fuse stopped at S, because jax don't support Slice op
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(6)
    ]
    chunk_slice = TensorSlice().new_chunk([None], None).data
    chunk_reduction = TensorSum(axis=(1,)).new_chunk([None], None).data
    graph = ChunkGraph([chunks[4], chunks[5]])
    list(map(graph.add_node, chunks[:6]))
    graph.add_node(chunk_slice)
    graph.add_edge(chunks[0], chunks[2])
    graph.add_edge(chunks[1], chunks[2])
    graph.add_edge(chunks[2], chunks[3])
    graph.add_edge(chunks[3], chunk_slice)
    graph.add_edge(chunk_slice, chunks[4])
    graph.add_edge(chunk_slice, chunks[5])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert fused_nodes[0].composed == chunks[:4]
    assert len(graph) == 4

    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        @                   @
          \               /
            @ --> @ --> @      ========>   Tail node count > 1, can't be fused.
          /               \
        @                   @

        fuse stopped at S, because jax don't support Slice op
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(7)
    ]
    graph = ChunkGraph([chunks[5], chunks[6]])
    list(map(graph.add_node, chunks[:7]))
    graph.add_edge(chunks[0], chunks[2])
    graph.add_edge(chunks[1], chunks[2])
    graph.add_edge(chunks[2], chunks[3])
    graph.add_edge(chunks[3], chunks[4])
    graph.add_edge(chunks[4], chunks[5])
    graph.add_edge(chunks[4], chunks[6])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert len(fused_nodes) == 0
    assert len(graph) == 7

    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        @           S       S
          \        /       /
            @ --> @ --> @      ========>   Tail node count > 1, can't be fused.
          /               \
        @                   @

        fuse stopped at S, because jax don't support Slice op
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(6)
    ]
    chunk_slices = [
        TensorSlice(_key=str(n)).new_chunk([None], None).data for n in range(2)
    ]
    graph = ChunkGraph([chunks[5], chunk_slices[0], chunk_slices[1]])
    list(map(graph.add_node, chunks[:6]))
    list(map(graph.add_node, chunk_slices[:2]))
    graph.add_edge(chunks[0], chunks[2])
    graph.add_edge(chunks[1], chunks[2])
    graph.add_edge(chunks[2], chunks[3])
    graph.add_edge(chunks[3], chunk_slices[0])
    graph.add_edge(chunks[3], chunks[4])
    graph.add_edge(chunks[4], chunks[5])
    graph.add_edge(chunks[4], chunk_slices[1])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert len(fused_nodes) == 0
    assert len(graph) == 8

    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        @
          \
            @
          /   \
        @      \
                 @   ========>   #
        @      /
          \   /
            @
          /
        @

        fuse stopped at S, because jax don't support Slice op
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(7)
    ]
    graph = ChunkGraph([chunks[6]])
    list(map(graph.add_node, chunks[:7]))
    graph.add_edge(chunks[0], chunks[2])
    graph.add_edge(chunks[1], chunks[2])
    graph.add_edge(chunks[3], chunks[5])
    graph.add_edge(chunks[4], chunks[5])
    graph.add_edge(chunks[2], chunks[6])
    graph.add_edge(chunks[5], chunks[6])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    sorted_composed = sorted(fused_nodes[0].composed, key=operator.attrgetter("key"))
    assert sorted_composed == chunks
    assert len(graph) == 1

    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        @
          \
            @
          /   \                            #
        @      \                              \
                 S --> @ --> @  ========>       S --> #
        @      /                              /
          \   /                            #
            @
          /
        @

        fuse stopped at S, because jax don't support Slice op
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(8)
    ]
    graph = ChunkGraph([chunks[7]])
    list(map(graph.add_node, chunks[:8]))
    graph.add_node(chunk_slice)
    graph.add_edge(chunks[0], chunks[2])
    graph.add_edge(chunks[1], chunks[2])
    graph.add_edge(chunks[3], chunks[5])
    graph.add_edge(chunks[4], chunks[5])
    graph.add_edge(chunks[2], chunk_slice)
    graph.add_edge(chunks[5], chunk_slice)
    graph.add_edge(chunk_slice, chunks[6])
    graph.add_edge(chunks[6], chunks[7])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert len(fused_nodes) == 3
    assert sorted(len(n.composed) for n in fused_nodes) == [2, 3, 3]
    assert len(graph) == 4
    assert graph.contains(chunk_slice)

    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        S
          \
            @
          /   \                         S
        @      \                           \
                 @ --- @   ========>    S --  #
        @      /     /                     /
          \   /     S                   S
            @
          /
        S

        fuse stopped at S, because jax don't support Slice op
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(6)
    ]
    chunk_slices = [
        TensorSlice(_key=str(n)).new_chunk([None], None).data for n in range(3)
    ]
    graph = ChunkGraph([chunks[5]])
    list(map(graph.add_node, chunks[:6]))
    list(map(graph.add_node, chunk_slices[:3]))
    graph.add_edge(chunk_slices[0], chunks[1])
    graph.add_edge(chunks[0], chunks[1])
    graph.add_edge(chunks[2], chunks[3])
    graph.add_edge(chunk_slices[1], chunks[3])
    graph.add_edge(chunks[1], chunks[4])
    graph.add_edge(chunks[3], chunks[4])
    graph.add_edge(chunks[4], chunks[5])
    graph.add_edge(chunk_slices[2], chunks[5])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert len(fused_nodes) == 1
    sorted_composed = sorted(fused_nodes[0].composed, key=operator.attrgetter("key"))
    assert sorted_composed == chunks
    assert len(graph) == 4
    assert graph.count_predecessors(fused_nodes[0]) == 3

    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        @ --> @ --> S --> @  ========>  # --> S --> @

        fuse stopped at S, because jax don't support Slice op
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(4)
    ]
    graph = ChunkGraph([chunks[2]])
    list(map(graph.add_node, chunks[:3]))
    graph.add_node(chunk_slice)
    graph.add_edge(chunks[0], chunks[1])
    graph.add_edge(chunks[1], chunk_slice)
    graph.add_edge(chunk_slice, chunks[2])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()

    print(fused_nodes[0].composed)
    print(chunks[:2])
    assert fused_nodes[0].composed == chunks[:2]
    assert len(fused_nodes) == 1

    r"""
        graph(@: node, S: Slice Chunk, #: fused_node):

        @ --> @ --> S --> @ --> @   ========>  # --> S --> #

        fuse stopped at S, because jax don't support Slice op
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(4)
    ]
    graph = ChunkGraph([chunks[3]])
    list(map(graph.add_node, chunks[:4]))
    graph.add_node(chunk_slice)
    graph.add_edge(chunks[0], chunks[1])
    graph.add_edge(chunks[1], chunk_slice)
    graph.add_edge(chunk_slice, chunks[2])
    graph.add_edge(chunks[2], chunks[3])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert fused_nodes[0].composed == chunks[:2]
    assert fused_nodes[1].composed == chunks[2:4]

    r"""
        graph(@: node, R: Reduction Chunk, #: fused_node):

        @ --> @ --> R --> @ --> @   ========>  # --> R --> #

        jax fuse will not fuse a reduction node
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(4)
    ]
    graph = ChunkGraph([chunks[3]])
    list(map(graph.add_node, chunks[:4]))
    graph.add_node(chunk_reduction)
    graph.add_edge(chunks[0], chunks[1])
    graph.add_edge(chunks[1], chunk_reduction)
    graph.add_edge(chunk_reduction, chunks[2])
    graph.add_edge(chunks[2], chunks[3])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert len(fused_nodes) == 2
    assert fused_nodes[0].composed == chunks[:2]
    assert fused_nodes[1].composed == chunks[2:4]
    assert len(graph) == 3

    r"""
        graph(@: node, R: Reduction Chunk, #: fused_node):

        R --> @ --> @   ========>  R --> #

        jax fuse will not fuse a reduction node
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(2)
    ]
    graph = ChunkGraph([chunks[1]])
    list(map(graph.add_node, chunks[:2]))
    graph.add_node(chunk_reduction)
    graph.add_edge(chunk_reduction, chunks[0])
    graph.add_edge(chunks[0], chunks[1])

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert len(fused_nodes) == 1
    assert fused_nodes[0].composed == chunks[:2]
    assert len(graph) == 2

    r"""
        graph(@: node, R: Reduction Chunk, #: fused_node):

        @ --> @ --> R   ========>  # --> R

        jax fuse will not fuse a reduction node
        """
    chunks = [
        TensorTreeAdd(args=[], _key=str(n)).new_chunk(None, None).data for n in range(2)
    ]
    graph = ChunkGraph([chunk_reduction])
    list(map(graph.add_node, chunks[:2]))
    graph.add_node(chunk_reduction)
    graph.add_edge(chunks[0], chunks[1])
    graph.add_edge(chunks[1], chunk_reduction)

    optimizer = JAXRuntimeOptimizer(graph)
    _, fused_nodes = optimizer.optimize()
    assert len(fused_nodes) == 1
    assert fused_nodes[0].composed == chunks[:2]
    assert len(graph) == 2
