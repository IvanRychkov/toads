import redis
from functools import wraps
import pickle


class RedisSerializer(redis.Redis):
    """Extended Redis class for working with serialized Python objects."""

    @wraps(redis.Redis.get)
    def get_serialized(self, name):
        try:
            return pickle.loads(super().get(name))
        except TypeError:
            return None

    @wraps(redis.Redis.set)
    def set_serialized(self, name, value, **kwargs):
        return super().set(name, pickle.dumps(value), **kwargs)


__all__ = ['RedisSerializer']
