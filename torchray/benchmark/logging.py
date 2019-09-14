# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module provides function that to be log information (e.g., benchmark
results) to a MongoDB database.

See :mod:`examples.standard_suite` for an example of how to use MongoDB for
logging benchmark results.

To start a MongoDB server, use

.. code:: shell

    $ python -m torchray.benchmark.server

"""
import io
import pickle

import bson
import numpy as np
import pymongo
import torch

from torchray.utils import get_config

__all__ = [
    'mongo_connect',
    'mongo_save',
    'mongo_load',
    'data_to_mongo',
    'data_from_mongo',
    'last_lines'
]

_MONGO_MAX_TRIES = 10


def mongo_connect(database):
    """
    Connect to MongoDB server and and return a
    :class:`pymongo.database.Database` object.

    Args:
        database (str): name of database.

    Returns:
        :class:`pymongo.database.Database`: database.
    """
    try:
        config = get_config()
        hostname = f"{config['mongo']['hostname']}:{config['mongo']['port']}"
        client = pymongo.MongoClient(hostname)
        client.server_info()
        database = client[database]
        return database
    except pymongo.errors.ServerSelectionTimeoutError as error:
        raise Exception(
            f"Cannot connect MonogDB at {hostname}") from error


def mongo_save(database, collection_key, id_key, data):
    """Save results to MongoDB database.

    Args:
        database (:class:`pymongo.database.Database`): MongoDB database to save
            results to.
        collection_key (str): name of collection.
        id_key (str): id key with which to store :attr:`data`.
        data (:class:`bson.binary.Binary` or dict): data to store in
            :attr:`db`.
    """
    collection = database[collection_key].with_options(
        write_concern=pymongo.WriteConcern(w=1))
    tries_left = _MONGO_MAX_TRIES
    while tries_left > 0:
        tries_left -= 1
        try:
            collection.replace_one(
                {'_id': id_key},
                data,
                upsert=True
            )
            return
        except (pymongo.errors.WriteConcernError, pymongo.errors.WriteError):
            if tries_left == 0:
                print(
                    f"Warning: could not write entry to mongodb after"
                    f" {_MONGO_MAX_TRIES} attempts."
                )
                raise


def mongo_load(database, collection_key, id_key):
    """Load data from MongoDB database.

    Args:
        database (:class:`pymongo.database.Database`): MongoDB database to save
            results to.
        collection_key (str): name of collection.
        id_key (str): id key to look up data.

    Returns:
        retrieved data (returns None if no data with :attr:`id_key` is found).
    """
    return database[collection_key].find_one({'_id': id_key})


def data_to_mongo(data):
    """Prepare data to be stored in a MongoDB database.

    Args:
        data (dict, :class:`torch.Tensor`, or :class:`np.ndarray`): data to
            prepare for storage in a MongoDB dataset (if dict, items are
            recursively prepared for storage). If the underlying data is
            not :class:`torch.Tensor` or :class:`np.ndarray`, then :attr:`data`
            is returned as is.

    Returns:
        :class:`bson.binary.Binary` or dict of :class:`bson.binary.Binary`:
        correctly formatted data to store in a MongoDB database.
    """
    if isinstance(data, dict):
        return {k: data_to_mongo(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        bytes_data = io.BytesIO()
        torch.save(data, bytes_data)
        bytes_data.seek(0)
        binary = bson.binary.Binary(bytes_data.read())
        return binary
    if isinstance(data, np.ndarray):
        return bson.binary.Binary(pickle.dumps(data, protocol=2),
                                  subtype=128)
    return data


def data_from_mongo(mongo_data, map_location=None):
    """Decode data stored in a MongoDB database.

    Args:
        mongo_data (:class:`bson.binary.Binary` or dict):
            data to decode (if dict, items are recursively decoded). If
            the underlying data type is not `:class:torch.Tensor` or
            something stored using :mod:`pickle`, then :attr:`mongo_data`
            is returned as is.
        map_location (function, :class:`torch.device`, str or dict): where to
            remap storage locations (see :func:`torch.load` for more details).
            Default: ``None``.

    Returns:
        decoded data.
    """

    if isinstance(mongo_data, dict):
        return {k: data_from_mongo(v) for k, v in mongo_data.items()}
    if isinstance(mongo_data, bson.binary.Binary):
        try:
            bytes_data = io.BytesIO(mongo_data)
            return torch.load(bytes_data, map_location=map_location)
        # If the underlying data is a numpy array, it throws a ValueError here.
        except Exception:
            pass
        try:
            return pickle.loads(mongo_data)
        except Exception:
            pass
    return mongo_data


def last_lines(string, num_lines):
    """Extract the last few lines from a string.

    The function extracts the last attr:`n` lines from the string attr:`str`.
    If attr:`n` is a negative number, then it extracts the first lines
    instead. It also skips lines beginning with ``'Figure('``.

    Args:
        string (str): string.
        num_lines (int): number of lines to extract.

    Returns:
        str: substring.
    """
    if string is None:
        return ''
    lines = string.strip().split('\n')
    lines = [l for l in lines if not l.startswith('Figure(')]
    if not lines:
        return ''
    if num_lines > 0:
        min_lines = min(num_lines, len(lines))
        lines_ = lines[-min_lines:]
        if num_lines < len(lines):
            lines_ = ['[...]'] + lines_
    if num_lines < 0:
        num_lines = -num_lines
        min_lines = min(num_lines, len(lines))
        lines_ = lines[:min_lines]
        if num_lines < len(lines):
            lines_ = lines_ + ['[...]']

    return '\n'.join(lines_)
