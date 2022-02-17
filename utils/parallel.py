from tqdm import tqdm
import ray

def parallel_compute_with_bar(computations) -> list:

    def to_iterator(obj_ids):
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])

    ret = []
    for x in tqdm(to_iterator(computations), total=len(computations)):
        ret.append(x)

    return ret