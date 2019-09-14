try:
    import importlib.resources as resources
except ImportError:
    import importlib_resources as resources

with resources.open_text('torchray', 'VERSION') as f:
    __version__ = f.readlines()[0].rstrip()
