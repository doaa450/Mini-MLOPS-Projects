import logging

from prodml.predict import DurationPredictor

logging.basicConfig(level=logging.INFO)

p = DurationPredictor().load()
sample = {"PULocationID": 82, "DOLocationID": 129, "trip_distance": 0.5}

print(p.predict_one(sample))
print(
    p.predict_batch(
        [sample, {"PULocationID": 226, "DOLocationID": 143, "trip_distance": 5.2}]
    )
)
