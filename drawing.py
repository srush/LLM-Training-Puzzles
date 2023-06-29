from typing import Sequence, Optional, FrozenSet
from lib import Model, Activation, Weight, ActivationGrad, WeightGrad

from chalk import rectangle, text, vcat, empty, hcat, Diagram, concat
from colour import Color

# backward_col = list(Color("Yellow").range_to("Red", LAYERS))


def draw(models : Sequence[Model]) -> Diagram:
    TIME = 2
    layers = models[0].LAYERS
    forward_col = list(Color("green").range_to("red", layers + 1))

    def square(layer:int, time:int, s: str="") -> Diagram:
        PAD = 0.2
        return (
            (
                rectangle(TIME * 0.92, 0.9).line_width(0)
                + text(s, 0.9 / ((len(s) + 1)/2))
                .translate(0, 0.1)
                .line_width(0.05)
                .fill_color(Color("black"))
            )
            .align_l()
            .translate(time * TIME, 0)
        )

    def draw_forward(e):
        return square(e.layer, e.time, ",".join(map(str, e.batches))).fill_color(
            forward_col[e.layer]
        )

    def draw_backward(e):
        return (
            square(e.layer, e.time, ",".join(map(str, e.batches)))
            .fill_color(forward_col[e.layer])
            .fill_opacity(0.2)
        )

    def draw_update(layer, time):
        return square(layer, time, "U").fill_color(Color("yellow")).fill_opacity(0.5)

    def draw_allred(layer, time):
        return square(layer, time).fill_color(forward_col[e.layer]).fill_opacity(0.5)

    rows = []

    MAXTIME = max(m.log[-1].time for m in models)

    # Time
    for gpu in range(len(models)):
        d = empty()
        box = (
            rectangle(TIME * (MAXTIME + 1), 1).fill_color(Color("lightgrey")).align_l()
        )
        box = box + text(str(gpu), 1).line_width(0).with_envelope(
            rectangle(TIME, 1)
        ).translate(-TIME, 0)
        d += box
        for e in models[gpu].log:
            if e.typ == "forward":
                d += draw_forward(e)
            if e.typ == "backward":
                d += draw_backward(e)
            if e.typ == "update":
                d += draw_update(e.layer, e.time)
            if e.typ in ["allreduce", "scatterreduce"]:
                d += draw_allred(e.layer, e.time)
        rows.append(d)
    d = vcat(rows)

    rows = []
    for gpu in range(len(models)):
        row = rectangle(TIME * (MAXTIME + 1), 1).fill_color(Color("white")).align_l()

        for e in models[gpu].log:
            can = (
                rectangle(TIME, e.memory / 100.0)
                .align_b()
                .align_l()
                .fill_color(Color("black"))
            )
            row = row.align_b() + can.translate(TIME * e.time, 0)
        rows.append(row)
    d2 = vcat(rows)
    d = vcat([d, d2])
    # return rectangle(1.5, 0.5) + d.scale_uniform_to_x(1).center_xy()
    return rectangle(1.2, 0.7) + d.scale_uniform_to_x(1).center_xy()


def draw_network(layers:int, weight: Optional[int]=None, before:int=-1, after:int=100, 
                 shards:FrozenSet[int]=frozenset({}), total:int=1, 
                 batches: FrozenSet[int]=frozenset({0}), total_batches=1, is_grad: bool=False) -> Diagram:
    forward_col = list(Color("green").range_to("red", layers+1))
    def layer(l: int) -> Diagram:
        W = 3
        H = 1 if l < layers else 0
        layer = rectangle(W, H).line_width(0.2).align_b()
        shard_h = H * (1 / total) 
        shard_w = W * (1 / total_batches)

        weight_shard = rectangle(shard_w, shard_h).line_width(0.01).align_t().align_l().line_color(Color("white"))
        weight_shards = concat([weight_shard.translate(batch * shard_w - (W/2), 
                                                       shard * shard_h - H) 
                                for shard in shards for batch in batches])
        
        connect_out = rectangle(1.5, 1.05).line_width(0.2).align_t()
        connect_w = 1.5 * (1 / total_batches)
        connect = rectangle(connect_w, 1).line_width(0.01).align_l().line_color(Color("white"))
        connect = concat([connect.translate(batch * connect_w - 1.5 * (1/2), 0.0) 
                                            for batch in batches]).align_t().translate(0, 0.025)

        if l == weight:
            weight_shards = weight_shards.fill_color(forward_col[l])
            if is_grad:
                weight_shards = weight_shards.fill_opacity(0.5)
        else:
            weight_shards = empty()
        if l == before: #or (after != 100 and l <= after):
            connect = connect.fill_color(forward_col[l]).fill_opacity(1)
        elif l == after + 1:
            connect = connect.fill_color(forward_col[l]).fill_opacity(0.5)
        else:
            connect = empty()
        base = connect_out + layer
        return  base, (connect + weight_shards).with_envelope(base)
    return vcat(reversed([layer(l)[0] for l in range(layers+1)])), vcat(reversed([layer(l)[1] for l in range(layers+1)]))


def draw_group(group):
    group = list(group.values())
    base = group[0].draw()[0]
    return base + concat([g.draw()[1] for g in group])

        
# hcat([base + Activation(2, 5, [0], 2).draw()[1], 
#       base + Weight(2, 5, 0, (0,), 2).draw()[1], 
#       base + WeightGrad(2, 5, {0}, 2, {1}, 2).draw()[1] + WeightGrad(2, 5, {1}, 2, {1}, 2).draw()[1],
#       base + ActivationGrad(2, 5, frozenset({1}), 2).draw()[1]], 1).render_svg("activation.svg", 500)