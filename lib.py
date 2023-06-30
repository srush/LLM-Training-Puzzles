from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, FrozenSet, TypeVar
from chalk import Diagram
import random

class Barrier:
    """Sync across n ranks"""
    def __init__(self, target: int):
        self.counter = 0
        self.target = target
        self.lock = asyncio.Lock()
        self.round = 0
        self.done = 0

    async def wait(self, rank: int) -> None:
        while self.done > 0:
            await asyncio.sleep(0.01)
        async with self.lock:
            self.counter += 1
        while self.counter < self.target:
            await asyncio.sleep(0.01)
        self.done += 1
        if rank == 0:
            await self.reset()

    async def reset(self) -> None:
        while self.done < self.target:
            await asyncio.sleep(0.01)
        self.counter = 0
        self.done = 0

T = TypeVar('T')
class Reduceable(Protocol[T]):
    """
    A type that can be reduced.
    """
    def __add__(self, other: T) -> T:
        ...

O = TypeVar('O')
class Gatherable(Protocol[O]):
    """
    A type that can be sharded.
    """
    def shard(self, shard: int, total: int) -> O:
        ...

    def is_complete(self) -> bool:
        ...

    def combine(self, other: O) -> O:
        ...

 
TO = TypeVar('TO')
class ReduceableGatherable(Reduceable[TO], Gatherable[TO]):
    pass


class Dist:
    def __init__(self, total: int) -> None:
        self.reduce: Optional[Any] = None
        self.gather: Optional[Any] = None
        self.ranks = total
        self.barrier = Barrier(total)
        self.queue : Sequence[asyncio.Queue[Any]] = [asyncio.Queue(maxsize=1) for i in range(total)]
        self.mtime = 0

    async def allreduce(self, rank: int, inp: T, time:int) -> Tuple[T, int]:
        if self.reduce is None:
            self.reduce = inp
        else:
            self.reduce = self.reduce + inp
        self.mtime = max(time, self.mtime)
        await self.barrier.wait(rank)
        q: T = self.reduce
        mtime = self.mtime
        await self.barrier.wait(rank)
        if rank == 0:
            self.reduce = None
            self.mtime = 0
        await self.barrier.wait(rank)
        return q, mtime

    async def allgather(self, rank: int, inp: O, time:int) -> Tuple[O, int]:
        if self.gather is None:
            self.gather = inp
        else:
            assert type(self.gather) == type(inp)
            self.gather = self.gather.combine(inp)
        self.mtime = max(time, self.mtime)
        await self.barrier.wait(rank)
        q: O = self.gather
        mtime = self.mtime
        await self.barrier.wait(rank)
        if rank == 0:
            self.gather = None
            self.mtime = 0
        await self.barrier.wait(rank)
        return q, mtime

    async def scatterreduce(self, rank: int, inp: TO, time:int) -> Tuple[TO, int]:
        x, time = await self.allreduce(rank, inp, time)
        y = x.shard(rank, self.ranks) # type: ignore
        return y, time # type: ignore

    async def receive(self, rank: int) -> Any:
        return await self.queue[rank].get()

    async def pass_to(self, rank: int, v: Any) -> None:
        await self.queue[rank].put(v)
        

@dataclass
class Weight(Gatherable["Weight"]):
    """
    The weights for a specific layer. Can be sharded.

    Required for forward and backward passes.
    """
    layer: int
    layers: int
    step: int
    shards: FrozenSet[int] = frozenset([0])
    total: int = 1
    
    def combine(self, other: Weight) -> Weight:
        return Weight(self.layer, self.layers, self.step, self.shards | other.shards, self.total)

    def memory(self) -> float:
        return (len(self.shards) / self.total) * HIDDEN * HIDDEN

    def shard(self, shard: int, total: int) -> Weight:
        assert self.is_complete()
        assert shard < total
        return Weight(self.layer, self.layers, self.step, frozenset([shard]), total)

    def is_complete(self) -> bool:
        return len(self.shards) == self.total

    def draw(self) -> Diagram:
        from drawing import draw_network
        return draw_network(self.layers, weight=self.layer, 
                            shards=self.shards, total=self.total)
    def _repr_svg_(self):
        d = self.draw()
        return (d[0] + d[1])._repr_svg_()

HIDDEN = 512
LENGTH = 256
@dataclass
class Activation:
    """
    Activations need for a specific layer for a specific set of batches.
    """
    layer: int
    layers: int
    batches: FrozenSet[int]
    total_batches: int

    def memory(self) -> int:
        return len(self.batches) * HIDDEN * LENGTH

    def draw(self) -> Diagram:
        from drawing import draw_network
        return draw_network(self.layers, before=self.layer, 
                            batches=self.batches, total_batches=self.total_batches)

    def _repr_svg_(self):
        d = self.draw()
        return (d[0] + d[1])._repr_svg_()

@dataclass
class WeightGrad(Reduceable["WeightGrad"], Gatherable["WeightGrad"]):
    """
    The gradient of the loss for a specific weight layer. 

    May be sharded to correspond to different parts of the weights. 

    May be split into different batches.
    """


    layer: int
    layers: int
    batches: FrozenSet[int]
    total_batches: int
    shards: FrozenSet[int] = frozenset([0])
    total: int = 1

    def __add__(self, other: WeightGrad) -> WeightGrad:
        assert self.layer == other.layer, "Only add same layer weight grads"
        assert self.shards == other.shards
        return WeightGrad(self.layer, self.layers, self.batches | other.batches, self.total_batches,
                           self.shards, self.total)

    def combine(self, other: WeightGrad) -> WeightGrad:
        return WeightGrad(self.layer, self.layers, self.batches, self.total_batches,
                          self.shards | other.shards, self.total)

    def memory(self) -> float:
        return (len(self.shards) / self.total) * HIDDEN * HIDDEN

    def shard(self, shard: int, total: int) -> WeightGrad:
        assert self.is_complete(), f"{self.shards} out of {self.total}"
        assert shard < total
        return WeightGrad(self.layer, self.layers, self.batches, self.total_batches, frozenset([shard]), total)

    def is_complete(self) -> bool:
        return len(self.shards) == self.total

    def draw(self) -> Diagram:
        from drawing import draw_network
        return draw_network(self.layers, weight=self.layer, shards=self.shards, 
                            batches=self.batches, 
                            total=self.total, total_batches=self.total_batches, is_grad=True)
    
    def _repr_svg_(self):
        d = self.draw()
        return (d[0] + d[1])._repr_svg_()


@dataclass
class OptState(Gatherable["OptState"]):
    """
    The state of the optimizer for a specific layer. Can be sharded.
    
    In pratice this represents ADAM's saved values needed for optimization. 

    Required for updating the weights.
    """

    layer: int
    layers: int
    step: int
    shards: FrozenSet[int] = frozenset([0,])
    total: int = 1

    def combine(self, other: OptState) -> OptState:
        return OptState(self.layer, self.layers, self.step, self.shards | other.shards, self.total)

    def memory(self) -> float:
        return HIDDEN * HIDDEN * (len(self.shards) / self.total)

    def draw(self) -> Diagram:
        from drawing import draw_network
        return draw_network(self.layers, before=self.layer, shards=self.shards, total=self.total)

    def _repr_svg_(self):
        d = self.draw()
        return (d[0] + d[1])._repr_svg_()

@dataclass
class ActivationGrad:
    """
    The gradient of the activations for a specific layer. 

    May be split into different batches.
    """

    layer: int
    layers: int
    batches: FrozenSet[int]
    total_batches: int

    def memory(self) -> int:
        return len(self.batches) * HIDDEN * LENGTH

    def draw(self) -> Diagram:
        from drawing import draw_network
        return draw_network(self.layers, after=self.layer, 
                            batches=self.batches, total_batches=self.total_batches)

    def _repr_svg_(self):
        d = self.draw()
        return (d[0] + d[1])._repr_svg_()


@dataclass
class Event:
    "Internal representations of events in the model for the visualizer"
    typ: str
    layer: Optional[int]
    rank: int
    time: int
    length: int
    memory: int
    batches: FrozenSet[int] = frozenset()


class Model:
    def __init__(self, rank: int=1, dist: Dist=Dist(1), layers: int=2, batches: int=1):
        self.rank = rank
        self.log: List[Event] = []
        self.dist = dist
        self.time = 0
        self.RANKS = dist.ranks
        self.LAYERS = layers
        self.BATCHES = batches
        self.final_weights: Dict[int, Weight] = {}
        
        self.weights: Dict[Any, Weight] = {}
        self.opt_states: Dict[Any, OptState] = {}
        self.activations: Dict[Any, Activation] = {}
        self.grad_activations: Dict[Any, ActivationGrad] = {}
        self.grad_weights: Dict[Any, WeightGrad] = {}

    def storage(self) -> Tuple[Dict[Any, Weight], Dict[Any, OptState], Dict[Any, Activation], Dict[Any, ActivationGrad], Dict[Any, WeightGrad]]:
        return self.weights, self.opt_states, self.activations, self.grad_activations, self.grad_weights

    def memory(self) -> int:
        mem = 0
        for d in list(self.storage()):
            assert isinstance(d, dict)
            for v in d.values():
                mem += v.memory()
        return mem

    def status(self):
        for d in list(self.storage()):
            for k, v in d.items():
                print(k, type(v), end=",")
        print()

    def event(self, typ: str, layer: Optional[int]=None, batches: FrozenSet[int]=frozenset({})) -> None:
        length = 0
        if typ in  ["loss", "allgather"]:
            length = 0
        if typ in ["forward", "backward"]:
            length = len(batches)
        if typ in ["update"]:
            length = 0.5
        if typ in ["allreduce", "scatterreduce", "allgather"]:
            length = 0.3
        if typ in ["pass"]:
            length = 0.2

        self.log.append(Event(typ, layer, self.rank, self.time, length, self.memory(), batches))
        self.time += length
    def load_weights(self, layer: int, shard: int = 0, total:int = 1 ) -> Tuple[Weight, OptState]:
        return Weight(layer, self.LAYERS, 0, frozenset([shard]), total),\
              OptState(layer, self.LAYERS, 0, frozenset([shard]), total)

    def set_final_weight(self, layer: int, weight:Weight) -> None:
        self.final_weights[layer] = weight

    def get_activation(self, batches: Sequence[int]) -> Activation:
        return Activation(0, self.LAYERS, frozenset(batches), self.BATCHES)

    def forward(self, layer: int, inp: Activation, weight: Weight) -> Activation:
        "Take in activation at layer i and return layer i + 1"
        self.event("forward", layer, inp.batches)
        assert weight.is_complete()
        assert weight.layer == layer, f"Weight should be layer {layer}"
        assert inp.layer == layer, f"Input should be layer {layer}"
        return Activation(layer + 1, self.LAYERS, inp.batches, self.BATCHES)

    def backward(
        self, layer: int, inp: Activation, grad: ActivationGrad, weight: Weight
    ) -> Tuple[WeightGrad, ActivationGrad]:
        self.event("backward", layer, inp.batches)
        assert weight.is_complete()
        assert weight.layer == layer, f"Weight should be layer {layer}"
        assert inp.layer == layer, f"Input should be layer {layer}"
        assert set(inp.batches) == set(
            grad.batches
        ), f"Batch mismatch {set(inp.batches)}"
        assert grad.layer == layer, f"Activation Grad should be layer {layer}"
        return (WeightGrad(layer, self.LAYERS, inp.batches, self.BATCHES), 
                ActivationGrad(layer - 1, self.LAYERS, inp.batches, self.BATCHES))

    def loss(self, inp: Activation) -> ActivationGrad:
        self.event("loss", self.LAYERS)
        assert inp.layer == self.LAYERS, f"Input should be final layer {self.LAYERS}"
        return ActivationGrad(self.LAYERS - 1, self.LAYERS, inp.batches, self.BATCHES)

    def update(self, layer: int, 
               weight_grad: WeightGrad,
               weight: Weight,
               opt_state: OptState,
               shard: int = 0) -> Tuple[Weight, OptState]:

        assert weight.layer == layer, f"Weight should be layer {layer}"
        assert weight_grad.layer == layer, f"Grad weight should be layer {layer}"
        assert set(weight_grad.batches) == set(
                range(self.BATCHES)
            ), f"{set(weight_grad.batches)}"
        assert opt_state.layer == layer
        if weight_grad.total > 1:
            assert weight.shards.issubset(weight_grad.shards), f"Weight {weight.shards}"
            assert opt_state.shards.issubset(weight_grad.shards),  f"Opt {opt_state.shards}"
        assert weight.step == opt_state.step
        new_opt = OptState(layer, self.LAYERS, opt_state.step + 1, opt_state.shards, opt_state.total)
        new_weight = Weight(layer, self.LAYERS, weight.step + 1, weight.shards, weight.total)
        self.event("update", None)
        return new_weight, new_opt
    
    def fake_grad(self, layer: int, batches= List[int]):
        return WeightGrad(layer, self.LAYERS, frozenset(batches), self.BATCHES)

    async def allreduce(self, v: T, layer: int) -> T:
        v, self.time = await self.dist.allreduce(self.rank, v, self.time)
        self.event("allreduce", layer)

        return v

    async def scatterreduce(self, v: TO, layer:int) -> TO:
        v, self.time = await self.dist.scatterreduce(self.rank, v, self.time)
        self.event("scatterreduce", layer)
        return v

    async def allgather(self, v: O, layer:int) -> O:
        v, self.time = await self.dist.allgather(self.rank, v, self.time)
        self.event("allgather", layer)
        return v

    async def pass_to(self, rank: int, v: Any) -> None:
        self.event("pass", None)
        await self.dist.pass_to(rank, (v, self.time))

    async def receive(self) -> Any:
        v, time = await self.dist.receive(self.rank)
        self.time = max(time, self.time)
        self.event("pass", None)
        return v

    @staticmethod
    def check(models : Sequence[Model]) -> None:
        for l in range(models[0].LAYERS):
            weight = None
            for m in models:
                if l in m.final_weights:
                    assert m.final_weights[l].step == 1
                    if weight is None:
                        weight = m.final_weights[l]
                    else:                   
                        weight = weight.combine(m.final_weights[l])
            assert weight is not None, f"Missing weight {l}"
            assert weight.is_complete(), f"Weight not complete {weight}"

        print("Correct!")
        from IPython.display import HTML
        pups = [
        "2m78jPG",
        "pn1e9TO",
        "MQCIwzT",
        "udLK6FS",
        "ZNem5o3",
        "DS2IZ6K",
        "aydRUz8",
        "MVUdQYK",
        "kLvno0p",
        "wScLiVz",
        "Z0TII8i",
        "F1SChho",
        "9hRi2jN",
        "lvzRF3W",
        "fqHxOGI",
        "1xeUYme",
        "6tVqKyM",
        "CCxZ6Wr",
        "lMW0OPQ",
        "wHVpHVG",
        "Wj2PGRl",
        "HlaTE8H",
        "k5jALH0",
        "3V37Hqr",
        "Eq2uMTA",
        "Vy9JShx",
        "g9I2ZmK",
        "Nu4RH7f",
        "sWp0Dqd",
        "bRKfspn",
        "qawCMl5",
        "2F6j2B4",
        "fiJxCVA",
        "pCAIlxD",
        "zJx2skh",
        "2Gdl1u7",
        "aJJAY4c",
        "ros6RLC",
        "DKLBJh7",
        "eyxH0Wc",
        "rJEkEw4"]
        return HTML("""
        <video alt="test" controls autoplay=1>
            <source src="https://openpuppies.com/mp4/%s.mp4"  type="video/mp4">
        </video>
        """%(random.sample(pups, 1)[0]))