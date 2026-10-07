from prodml.logging_conf import correlation_id_var, setup_logging
from prodml.predict import DurationPredictor

setup_logging()
correlation_id_var.set("test-123")

p = DurationPredictor().load()
sample = {"PULocationID": 82, "DOLocationID": 129, "trip_distance": 0.5}
far = {"PULocationID": 226, "DOLocationID": 143, "trip_distance": 150}

print(p.predict_one(sample))
print(p.predict_batch([sample, far]))
