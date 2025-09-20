import numpy as np

class TPlan:
    ct : np.ndarray = None
    voi : np.ndarray = None
    voinames : np.ndarray = None
    voxelsize : float = None
    filepath : str = None

    def __init__(self, **kwargs):
        """Initialize the container with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        """String representation of the container."""
        items = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"TPlan({', '.join(items)})"

    def __repr__(self):
        """Developer-friendly representation."""
        return self.__str__()

    @classmethod
    def from_dict(cls, data_dict):
        """Create a DataContainer instance from a dictionary."""
        return cls(**data_dict)

    def get(self, key, default=None):
        """Get an attribute with a default value if it doesn't exist."""
        return getattr(self, key, default)

    def set(self, key, value):
        """Set an attribute."""
        setattr(self, key, value)

    def to_dict(self):
        """Convert the container to a dictionary."""
        return self.__dict__.copy()