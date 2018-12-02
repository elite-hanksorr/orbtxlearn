import io
import pickle
import zlib

import numpy as np
import sqlite3

def dict_factory(cur, row: list) -> dict:
    return {col[0]: row[i] for i, col in enumerate(cur.description)}

# https://stackoverflow.com/a/46358247/1529586
def adapt_array(arr: np.ndarray):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return zlib.compress(out.read())

# https://stackoverflow.com/a/46358247/1529586
def convert_array(data) -> np.ndarray:
    return np.load(io.BytesIO(zlib.decompress(data)))

def sqlite_register_custom_types():
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter('ndarray', convert_array)

    sqlite3.register_adapter(dict, lambda d: pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL))
    sqlite3.register_converter('dict', lambda bs: pickle.loads(bs))