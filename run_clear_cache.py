from diskcache import Cache

cache_1 = Cache(".cachedir/feature_selection")
cache_2 = Cache(".cachedir/data")
cache_1.clear()
cache_2.clear()
