"""
File: solutionset_1.py
Author: Ruben Bosschaert
Creation date: 20 Sept 2025
Description: This script provides the solution to problemset_1,
providing a function which displays the CT image together with the contour of the tumor,
the spinal chord and the esophagus
"""

from typing import Union, List, Dict, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.helpers import mat_loader

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
        """String representation of the TPlan."""
        items = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"TPlan({', '.join(items)})"

    def __repr__(self):
        """Developer-friendly representation."""
        return self.__str__()

    @classmethod
    def from_dict(cls, data_dict):
        """Create a TPlan instance from a dictionary."""
        return cls(**data_dict)


def load_tp_plan_data(tp_plan_path: Union[Path | str]) -> TPlan:
    """Loads data with the mat_loader and stores the data in a returned TPlan object"""
    if not Path(tp_plan_path).exists():
        print(f'Filepath to load tp plan data from does not exist: {tp_plan_path}')
        raise Exception
    else:
        print(f"Loading tp plan data from path :{tp_plan_path}")
        tp_plan_dict = mat_loader.loadmat(str(tp_plan_path))
        print(f"Succes! Dictionary contents: {tp_plan_dict}")

        # add filepath to the dictionary
        tp_plan_dict['TPlan']['filepath'] = tp_plan_path

        # load dictonary into TPlan object/data_model
        tp_plan_obj = TPlan.from_dict(tp_plan_dict['TPlan'])
        print(f"Loaded the dictionary data into a TPlan object: {tp_plan_obj}")

        return tp_plan_obj

def get_voinames_number(all_voinames: List|np.ndarray)-> Dict[str,int]:
    """Create a mapping of voi names to sequential numbers starting from 1."""
    if len(all_voinames) == 0 or not all(isinstance(item, str) for item in all_voinames):
        raise TypeError("all_voinames must be a non-empty list of strings")
    else:
        return {voiname: number for number, voiname in enumerate(all_voinames, 1)}


def plot_voi_contour(contour_array: np.ndarray, voinames_numbered: Dict[str, int], voiname_to_plot: str, color: str)->np.ndarray[int]:
    """Plot the contour array for the voiname to plot name with the indicated color.
    The mapping for the voiname and the mask number in the contour array is provided in the voinames numbered dictionary"""

    # Check if the VOI name exists in the dictionary
    if voiname_to_plot not in voinames_numbered:
        raise ValueError(f"VOI name '{voiname_to_plot}' not found in voinames_numbered dictionary")

    # Get the mask number for the specified VOI
    mask_number = voinames_numbered[voiname_to_plot]

    # Create a binary mask for the specific VOI
    voi_mask = (contour_array == mask_number)

    # Plot the contour
    plt.contour(voi_mask, levels=[0.5], colors=[color], linewidths=2)

    # If we want filled contours instead of just outlines
    # plt.contourf(voi_mask, levels=[0.5, 1], colors=[color], alpha=0.3)

    # Add legend entry
    plt.plot([], [], color=color, linewidth=2, label=voiname_to_plot)

    return voi_mask


def visualize_tp_plan_data(tp_plan_path: Union[Path | str], voinames_colors_visualization: List[Tuple[str,str]]) :
    """Visualize the tp plan data using matplotlib by showing the ct image data and plotting the contours over it"""
    # Get TPlan object
    tp_plan_obj = load_tp_plan_data(tp_plan_path)

    # Link the names of the vois to the voi fields coding numbers (luckily related to the order of the names)
    voinames_numbered_dict = get_voinames_number(tp_plan_obj.voinames)

    # Plot the ct imaging data
    plt.imshow(tp_plan_obj.ct,  cmap='grey')

    # Plot the contour data
    voi_masks = {}
    for voiname, color in voinames_colors_visualization:
        voi_mask = plot_voi_contour(tp_plan_obj.voi,voinames_numbered_dict,voiname,color)
        voi_masks[f'{voiname}'] = voi_mask

    plt.title('TPlan with contours')

    #Add legend
    plt.legend(fontsize=8)
    plt.show()

if __name__ == '__main__':
    visualize_tp_plan_data(
        tp_plan_path=Path(r'H:\_KlinFysica\_RT\phys_med_RT_planning\treatment_planning_2025_uv\utils\data\patientdata.mat'),
        voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','blue')])