from lib import *
async def basic(model):
    model.activations[0] = model.get_activation(range(model.BATCHES))
    model.load_weights(range(model.LAYERS))

    # Forward
    for l in range(model.LAYERS):
        model.activations[l+1] = model.forward(l, model.activations[l])

    # Backward
    model.grad_activations[model.LAYERS] = model.loss(model.activations[model.LAYERS])

    for l in range(model.LAYERS-1, -1, -1):
        model.grad_weights[l], model.grad_activations[l] = model.backward(l, model.activations[l], model.grad_activations[l+1])
        del model.grad_activations[l+1], model.activations[l]

    # Update
    model.update(list(range(model.LAYERS)), model.grad_weights)

    return [model]

async def ddp(model):
    model.activations[0] = model.get_activation([model.rank])
    model.load_weights(range(model.LAYERS))

    # Forward
    for l in range(model.LAYERS):
        model.activations[l+1] = model.forward(l, model.activations[l])

    # Backward
    model.grad_activations[model.LAYERS] = model.loss(model.activations[model.LAYERS])

    for l in range(model.LAYERS-1, -1, -1):
        model.grad_weights[l], model.grad_activations[l] = model.backward(l, model.activations[l], model.grad_activations[l+1])
        del model.grad_activations[l+1], model.activations[l]

    # Update
    model.grad_weights = await model.allreduce(model.grad_weights)
    model.update(list(range(model.LAYERS)), model.grad_weights)

    return model




async def pipeline(model):
    per_rank = model.LAYERS // model.RANKS
    my_layers = list([l + (model.rank * per_rank) for l in range(per_rank)])
    model.load_weights(my_layers)

    if model.rank == 0:
        model.activations[0] = model.get_activation(range(model.BATCHES))
    else:
        model.activations[my_layers[0]] = await model.receive()

    # Forward
    for l in my_layers:
        model.activations[l+1] = model.forward(l, model.activations[l])

    # Backward
    if model.rank == model.RANKS - 1:
        model.grad_activations[model.LAYERS] = model.loss(model.activations[model.LAYERS])
    else:
        await model.pass_to(model.rank + 1, model.activations[l + 1])
        model.grad_activations[l + 1] = await model.receive()

    for l in reversed(my_layers):
        model.grad_weights[l], model.grad_activations[l] = model.backward(l, model.activations[l], model.grad_activations[l+1])
        del model.grad_activations[l+1], model.activations[l]

    if model.rank != 0:
        await model.pass_to(model.rank - 1, model.grad_activations[l])

    # Update
    model.update(my_layers, model.grad_weights)
    return model

async def gpipe(model):
    per_rank = model.LAYERS // model.RANKS
    my_layers = list([l + (model.rank * per_rank) for l in range(per_rank)])
    model.load_weights(my_layers)

    for mb in [0, 1]:
        # Forward
        if model.rank == 0:
            model.activations[0, mb] = model.get_activation([mb])
        else:
            model.activations[my_layers[0], mb] = await model.receive()

        for l in my_layers:
            model.activations[l+1, mb] = model.forward(l, model.activations[l, mb])
        if model.rank != model.RANKS - 1:
            await model.pass_to(model.rank + 1, model.activations[l + 1, mb])

    for mb in [0, 1]:
        # Backward
        if model.rank == model.RANKS - 1:
            model.grad_activations[model.LAYERS, mb] = model.loss(model.activations[model.LAYERS, mb])
        else:
            model.grad_activations[my_layers[-1] + 1, mb] = await model.receive()

        for l in reversed(my_layers):
            model.grad_weights[l, mb], model.grad_activations[l, mb] = \
                model.backward(l, model.activations[l, mb], model.grad_activations[l+1, mb])
            del model.grad_activations[l+1, mb], model.activations[l, mb]

        if model.rank != 0:
            await model.pass_to(model.rank - 1, model.grad_activations[l, mb])

    # Update
    for l in reversed(my_layers):
        model.grad_weights[l] = model.grad_weights[l, 0] + model.grad_weights[l, 1]
        del  model.grad_weights[l, 0], model.grad_weights[l, 1]
    model.update(my_layers, model.grad_weights)
    return model



async def run():
    model1 = await basic(Model(0, Dist(1), layers=6, batches=2))
    Model.check(model1)

    async def main():
        ranks = 2
        dist = Dist(ranks)
        return await asyncio.gather(*[ddp(Model(rank, dist, layers=6, batches=2))
                                      for rank in range(ranks)])
    model2 = await main()
    Model.check(model2)


    async def main():
        ranks = 4
        dist = Dist(ranks)
        return await asyncio.gather(*[pipeline(Model(rank, dist, layers=8, batches=2))
                                      for rank in range(ranks)])

    model3 = await main()
    Model.check(model3)


    async def main():
        ranks = 4
        dist = Dist(ranks)
        return await asyncio.gather(*[gpipe(Model(rank, dist, layers=8, batches=2))
                                      for rank in range(ranks)])

    model4 = await main()
    Model.check(model4)

    set_svg_height(600)
    set_svg_draw_height(100)

    vcat([draw(model1), draw(model2), draw(model3), draw(model4)], 0.1).render_svg("out.svg")
asyncio.run(run())


