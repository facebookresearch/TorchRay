# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module is used to start and run a MongoDB server.

To start a MongoDB server, use

.. code:: shell

    $ python -m torchray.benchmark.server

"""
import subprocess
from torchray.utils import get_config


def run_server():
    """Runs an instance of MongoDB as a logging server."""
    config = get_config()
    command = [
        config['mongo']['server'],
        '--dbpath', config['mongo']['database'],
        '--bind_ip', config['mongo']['hostname'],
        '--port', str(config['mongo']['port'])
    ]
    print(f"Command: {' '.join(command)}.")
    code = subprocess.call(command, cwd=".")
    print(f"Return code {code}")


if __name__ == '__main__':
    run_server()
