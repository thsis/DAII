"""Script that generates config-jsons."""
import os
import json

learn = [1, 0.5, 0.01]
nodes = [1, 10, 20]
epochs = [100, 1000, 10000]
architectures = ["nn1l", "nn2l", "nn5l"]

i = 0
for a in architectures:
    for l in learn:
        for n in nodes:
            for e in epochs:
                i += 1
                if a == "nn1l":
                    architecture = n
                elif a == "nn2l":
                    architecture = (28, n, n)
                elif a == "nn5l":
                    architecture = (28, n, n, n, n, n)

                name = os.path.join("configs", "config" + str(i))

                config = {"logdir": os.path.join("config", a, a+"_"+str(i)),
                          "epochs": e,
                          "nodes": architecture}

                with open(name, 'w') as f:
                    json.dump(config, f)
