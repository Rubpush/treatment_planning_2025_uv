"""
File: solutionset_1.py
Author: Ruben Bosschaert
Creation date: 20 Sept 2025
Description: This script provides the solution to problemset_1,
providing a script which displays the CT image together with the contour of the tumor,
the spinal chord and the esophagus.

To perform this, run visualize_tp_plan_data with input arguments
tp_plan_path and voinames_colors_visualization which are also indicated in the example and if __name__ == '__main__':
part at the end of this script. """

from typing import Union, List, Dict, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.helpers import mat_loader

class TPlan:
    filepath: str = None
    ct : np.ndarray = None
    dosegrid : np.ndarray = None
    voi : np.ndarray = None
    voinames : np.ndarray = None
    voxelsize : float = None
    """
    Treatment plan container that supports both predefined and dynamic attributes.

    Predefined attributes:
        filepath, ct, dosegrid, voi, voinames, voxelsize

    Additional attributes can be added dynamically via the constructor or from_dict().
    """


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

def load_dose_data(dose_path: Union[Path | str], tplan:TPlan) -> TPlan:
    """Loads data with the mat_loader and stores the data in a returned TPlan object"""
    if not Path(dose_path).exists():
        print(f'Filepath to load dose data from does not exist: {dose_path}')
        raise Exception
    else:
        print(f"Loading dose data from path :{dose_path}")
        dose_path_dict = mat_loader.loadmat(str(dose_path))
        print(f"Succes! Dictionary contents: {dose_path}")

        # add dose to TPlan object
        tplan.__setattr__('dosegrid',dose_path_dict['dose'])

        return tplan

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


def visualize_tp_plan_data(tp_plan_path: Union[Path | str], voinames_colors_visualization: List[Tuple[str,str]],
                           show_plot:bool =True):
    """Visualize treatment planning data showing CT image with overlaid VOI contours.

    Args:
        tp_plan_path (Union[Path, str]): Path to treatment plan data file (.mat format).
        voinames_colors_visualization (List[Tuple[str, str]]): List of (voi_name, color) tuples.
            VOI names must match those in the data file. Colors should be valid matplotlib specs.
        show_plot (bool): argument to show plot or not, if not shown, returns the plot, else returns none

    Returns:
        None: Displays plot with plt.show().

    Raises:
        ValueError: If inputs are None/empty or voinames_colors_visualization has wrong format.

    Example:
        >>> visualize_tp_plan_data(Path('data.mat'), [('tumor', 'red'), ('esophagus', 'green')],True)
    """
    if not tp_plan_path or not voinames_colors_visualization:
        raise ValueError(f"Missing input arguments for tp_plan_path or voinames_colors_visualization")

    # Check if voinames_colors_visualization is a list of two string tuples
    elif not (isinstance(voinames_colors_visualization, list) and
            all(isinstance(item, tuple) and len(item) == 2 and
                all(isinstance(x, str) for x in item) for item in voinames_colors_visualization)):
        raise ValueError (f"Error: Expected list of tuples with 2 strings each, got: {type(voinames_colors_visualization).__name__}")

    # Get TPlan object
    tp_plan_obj = load_tp_plan_data(tp_plan_path)

    # Link the names of the vois to the voi fields coding numbers (luckily related to the order of the names)
    voinames_numbered_dict = get_voinames_number(tp_plan_obj.voinames)

    # Plot the ct imaging data
    ct_img = plt.imshow(tp_plan_obj.ct,  cmap='grey', origin='lower')

    # Plot the contour data
    voi_masks = {}
    for voiname, color in voinames_colors_visualization:
        voi_mask = plot_voi_contour(tp_plan_obj.voi,voinames_numbered_dict,voiname,color)
        voi_masks[f'{voiname}'] = voi_mask


    #Add legend
    plt.legend(fontsize=8)

    if show_plot:
        plt.title('TPlan with contours')

        plt.show()
        return None
    else:
        return tp_plan_obj, ct_img

def plot_dose_on_ct(tp_plan_path: Union[Path | str], voinames_colors_visualization: List[Tuple[str,str]],
                    dose_path:Union[Path | str], show_plot: bool = True) :
    """Visualize treatment planning data showing CT image with overlaid VOI contours.

    Args:
        tp_plan_path (Union[Path, str]): Path to treatment plan data file (.mat format).
        voinames_colors_visualization (List[Tuple[str, str]]): List of (voi_name, color) tuples.
            VOI names must match those in the data file. Colors should be valid matplotlib specs.
        dose_path (Union[Path, str]): Path to dose data file (.mat format).
        show_plot (bool): argument to show plot or not, if not shown, returns the plot, else returns none

    Returns:
        None: Displays plot with plt.show().

    Raises:
        ValueError: If inputs are None/empty or voinames_colors_visualization has wrong format.

    Example:
        >>> plot_dose_on_ct(Path('ctdata.mat'), [('tumor', 'red'), ('esophagus', 'green')],Path('dosedata.mat'),True)
    """
    tp_plan_obj, ct_img = visualize_tp_plan_data(
        tp_plan_path=tp_plan_path,
        voinames_colors_visualization=voinames_colors_visualization,
        show_plot=False)

    # Update tplan with dose data
    tp_plan_obj = load_dose_data(dose_path, tp_plan_obj)
    plt.imshow(tp_plan_obj.dosegrid,alpha=0.4,cmap='jet', origin='lower')
    if show_plot:
        plt.title('TPlan with contours and dose')

        plt.show()
        return None

if __name__ == '__main__':
    # visualize_tp_plan_data(
    #     tp_plan_path=Path(r'H:\_KlinFysica\_RT\phys_med_RT_planning\treatment_planning_2025_uv\utils\data\patientdata.mat'),
    #     voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','blue')],
    #     show_plot=True)

    plot_dose_on_ct(tp_plan_path=r'H:\_KlinFysica\_RT\phys_med_RT_planning\treatment_planning_2025_uv\utils\data\patientdata.mat',
                    voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','blue')],
                    dose_path=r'H:\_KlinFysica\_RT\phys_med_RT_planning\treatment_planning_2025_uv\utils\data\exampledose.mat',
                    show_plot=True,)