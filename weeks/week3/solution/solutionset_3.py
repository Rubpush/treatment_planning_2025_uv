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
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from scipy import ndimage


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

def project_root_provider()->Path:
    """Provides the root path of the project"""
    return Path(__file__).parent.parent.parent.parent

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

        # add dose data to TPlan object
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
        visualize_tp_plan_data(Path('data.mat'), [('tumor', 'red'), ('esophagus', 'green')],True)
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


def plot_dose_visualization(tp_plan_obj, dose_cutoff: float = 2.0,
                            viz_mode: str = 'gradient',
                            dose_levels: List[float] = None,
                            n_intervals: int = 10,
                            alpha=0.3):
    """
    Plot dose data with three different visualization options.

    Args:
        tp_plan_obj: TPlan object containing dose data
        dose_cutoff: Minimum dose value to display
        viz_mode: 'gradient', 'isodose', or 'intervals'
        dose_levels: Custom dose levels for isodose lines (optional)
        n_intervals: Number of intervals for 'intervals' mode
    """

    if viz_mode not in ['gradient', 'isodose', 'intervals']:
        raise ValueError("viz_mode must be 'gradient', 'isodose', or 'intervals'")

    # Apply dose cutoff
    dose_masked = np.ma.masked_where(tp_plan_obj.dosegrid < dose_cutoff, tp_plan_obj.dosegrid)

    if viz_mode == 'gradient':
        # Standard gradient dose display
        dose_img = plt.imshow(dose_masked, alpha=alpha, cmap='jet', origin='lower')
        plt.colorbar(dose_img, label='Dose (Gy)')

    elif viz_mode == 'isodose':
        # Smooth isodose lines display
        def visualize_isodose_contours(tp_plan_obj, dose_cutoff, dose_levels, sigma=0.5, alpha=0.7, linewidth=0.8,dose_levels_step=10):
            """ Visualize isodose contours on the dose grid."""

            # Smooth the data by taking a weighted average of a region for better contours (although 0.5 is low smoothing)
            dose_smooth = ndimage.gaussian_filter(tp_plan_obj.dosegrid, sigma=sigma)

            # Find the highest dose value in the dose grid for color bar and scaling
            max_dose = np.nanmax(tp_plan_obj.dosegrid)

            # Define dose levels for contours
            if dose_levels is None:
                dose_levels = np.linspace(dose_cutoff, max_dose, dose_levels_step)
            elif max(dose_levels)<max_dose:
                dose_levels.append(max_dose)

            # Create filled contours for background
            contourf = plt.contourf(dose_smooth, levels=dose_levels, cmap='jet', alpha=alpha, origin='lower')

            # Add contour lines
            contours = plt.contour(dose_smooth, levels=dose_levels, colors='black',
                                   linewidths=linewidth, alpha=alpha, origin='lower')

            plt.colorbar(contourf, label='Dose (Gy)')

        visualize_isodose_contours(tp_plan_obj, dose_cutoff, dose_levels, sigma=0.5, alpha=0.7, linewidth=0.8)

    elif viz_mode == 'intervals':
        def visualize_dose_intervals(tp_plan_obj, dose_cutoff, n_intervals):
            """ Visualize dose with discrete dose intervals."""
            # Discrete dose intervals with single colors
            max_dose = np.nanmax(tp_plan_obj.dosegrid)

            # Create dose intervals from cutoff to max dose with given intervals
            dose_bins = np.linspace(dose_cutoff, max_dose, n_intervals + 1)

            # Digitize/bin dose values into intervals
            dose_binned = np.digitize(tp_plan_obj.dosegrid, dose_bins)

            # Mask/remove values below chosen cutoff
            dose_binned_masked = np.ma.masked_where(tp_plan_obj.dosegrid < dose_cutoff, dose_binned)

            # Get discrete colors from jet colormap
            colors = cm.jet(np.linspace(0, 1, n_intervals))
            discrete_cmap = ListedColormap(colors)

            # Plot the binned dose data with discrete colormap
            dose_img = plt.imshow(dose_binned_masked, cmap=discrete_cmap,
                                  alpha=0.6, origin='lower',
                                  vmin=0.5, vmax=n_intervals + 0.5)

            # Create custom colorbar with centered interval labels
            cbar = plt.colorbar(dose_img, label='Dose (Gy)')

            # Position ticks at the center of each color band
            tick_positions = np.arange(1, n_intervals + 1)
            cbar.set_ticks(tick_positions)

            # Create interval labels
            interval_labels = [f'{dose_bins[i]:.0f}-{dose_bins[i + 1]:.0f}'
                               for i in range(len(dose_bins) - 1)]
            cbar.set_ticklabels(interval_labels)

        visualize_dose_intervals(tp_plan_obj, dose_cutoff, n_intervals)

    else:
        raise ValueError("viz_mode must be 'gradient', 'isodose', or 'intervals'")


def plot_dose_on_ct(tp_plan_path: Union[Path, str],
                    voinames_colors_visualization: List[Tuple[str, str]],
                    dose_path: Union[Path, str],
                    dose_cutoff: float = 5.0,
                    viz_mode: str = 'gradient',
                    dose_levels: List[float] = None,
                    n_intervals: int = 10,
                    show_plot: bool = True):
    """
    Enhanced dose visualization with multiple display options.

    Args:
        tp_plan_path: Path to treatment plan data
        voinames_colors_visualization: List of (voi_name, color) tuples
        dose_path: Path to dose data
        dose_cutoff: Minimum dose to display
        viz_mode: 'gradient', 'isodose', or 'intervals'
        dose_levels: Custom dose levels for isodose (optional)
        n_intervals: Number of discrete intervals for 'intervals' mode
        show_plot: Whether to display the plot
    """

    # Load your existing data (assuming these functions exist)
    tp_plan_obj, ct_img = visualize_tp_plan_data(
        tp_plan_path=tp_plan_path,
        voinames_colors_visualization=voinames_colors_visualization,
        show_plot=False)

    tp_plan_obj = load_dose_data(dose_path, tp_plan_obj)

    # Apply the selected visualization mode
    plot_dose_visualization(tp_plan_obj, dose_cutoff, viz_mode, dose_levels, n_intervals)

    if show_plot:
        title_suffix = {'gradient': 'Gradient', 'isodose': 'Isodose Lines', 'intervals': 'Discrete Intervals'}
        plt.title(f'TPlan with contours and dose - {title_suffix[viz_mode]}')
        plt.show()
        return None

    return tp_plan_obj

def calculate_raddepth(angle) ->np.ndarray:
    """Calculate the radial depth for a given angle."""
    # Convert angle from degrees to radians
    theta = np.radians(angle)

    # Calculate radial depth using the formula
    raddepth = 10 / np.cos(theta)

    return raddepth

if __name__ == '__main__':
    # # CT with contours visualization
    # visualize_tp_plan_data(
    #     tp_plan_path=Path(r'H:\_KlinFysica\_RT\phys_med_RT_planning\treatment_planning_2025_uv\utils\data\patientdata.mat'),
    #     voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','blue')],
    #     show_plot=True)

    # # Mode 1: Dose visualization - Gradient
    # plot_dose_on_ct(tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #                 voinames_colors_visualization=[('tumor', 'purple'),('esophagus','brown'),('spinal cord','magenta')],
    #                 dose_path=str(project_root_provider()) + r'.\utils\data\exampledose.mat',
    #                 viz_mode='gradient',
    #                 dose_cutoff=5.0,
    #                 show_plot=True
    # )

    # # Mode 2: Dose visualization - Isodose lines
    # plot_dose_on_ct(tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #                 voinames_colors_visualization=[('tumor', 'purple'),('esophagus','brown'),('spinal cord','magenta')],
    #                 dose_path=str(project_root_provider()) + r'.\utils\data\exampledose.mat',
    #                 viz_mode='isodose',
    #                 dose_cutoff=5.0,
    #                 dose_levels=[5, 10, 20, 30, 40, 50, 60, 70],  # Custom dose levels in Gy
    #                 show_plot=True
    # )

    # # Mode 3: Dose visualization - Discrete intervals
    # plot_dose_on_ct(tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #                 voinames_colors_visualization=[('tumor', 'purple'),('esophagus','brown'),('spinal cord','magenta')],
    #                 dose_path=str(project_root_provider()) + r'.\utils\data\exampledose.mat',
    #                 viz_mode='intervals',
    #                 dose_cutoff=5.0,
    #                 n_intervals=8,  # Number of discrete dose bands
    #                 show_plot=True
    #                 )