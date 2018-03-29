"""Script that generates config-jsons."""
import os
import json

learn = [0.005, 0.01]
nodes = [1, 10, 20]
epochs = [10000, 100000]
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

                name = os.path.join("configs", a, "config" + str(i) + ".json")

                config = {
                    "name": a + str(i),
                    "init": {"units": architecture},
                    "train": {
                        "learn": l,
                        "logdir": os.path.join("models", a, a+"_"+str(i)),
                        "epochs": e}}

                with open(name, 'w') as f:
                    json.dump(config, f)
