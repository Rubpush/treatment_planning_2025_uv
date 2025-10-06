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


def visualize_tp_plan_data(tp_plan_path: Union[Path | str], voinames_colors_visualization: List[Tuple[str,str]] = None,
                           show_plot:bool =True):
    """Visualize treatment planning data showing CT image with overlaid VOI contours.

    Args:
        tp_plan_path (Union[Path, str]): Path to treatment plan data file (.mat format).
        voinames_colors_visualization (List[Tuple[str, str]]): List of (voi_name, color) tuples.
            VOI names must match those in the data file. Colors should be valid matplotlib specs.
            If none, no contours will be plotted
        show_plot (bool): argument to show plot or not, if not shown, returns the plot, else returns none

    Returns:
        None: Displays plot with plt.show().

    Raises:
        ValueError: If inputs are None/empty or voinames_colors_visualization has wrong format.

    Example:
        visualize_tp_plan_data(Path('data.mat'), [('tumor', 'red'), ('esophagus', 'green')],True)
    """
    if not tp_plan_path:
        raise ValueError(f"Missing input arguments for tp_plan_path or voinames_colors_visualization")

    # Check if voinames_colors_visualization is a list of two string tuples or None
    elif not voinames_colors_visualization is None and not (isinstance(voinames_colors_visualization, list) and
            all(isinstance(item, tuple) and len(item) == 2 and
                all(isinstance(x, str) for x in item) for item in voinames_colors_visualization)):
        raise ValueError (f"Error: Expected list of tuples with 2 strings each, got: {type(voinames_colors_visualization).__name__}")

    # Get TPlan object
    tp_plan_obj = load_tp_plan_data(tp_plan_path)

    # Link the names of the vois to the voi fields coding numbers (luckily related to the order of the names)
    voinames_numbered_dict = get_voinames_number(tp_plan_obj.voinames)

    # Plot the ct imaging data
    ct_img = plt.imshow(tp_plan_obj.ct,  cmap='grey', origin='lower')

    # Plot the contour data if
    if voinames_colors_visualization:
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


def get_body_mask(tp_plan_obj:TPlan)-> np.ndarray:
    # Link the names of the vois to the voi fields coding numbers (luckily related to the order of the names)
    voinames_numbered_dict = get_voinames_number(tp_plan_obj.voinames)

    # Initialize a body mask object with the same size as the original mask
    body_mask = np.zeros_like(tp_plan_obj.voi, dtype=bool)

    # Iterate through all VOIs and add these to the body mask
    for voi_name, voi_number in voinames_numbered_dict.items():
        if voi_name not in ['background', 'air']:  # filter as needed
            # Create mask for this VOI
            voi_mask = (tp_plan_obj.voi == voi_number)
            # Add to body mask using logical OR
            body_mask = body_mask | voi_mask

    return body_mask


def get_isocenter(tp_plan_obj:TPlan, isocenter_method):
    # Link the names of the vois to the voi fields coding numbers (luckily related to the order of the names)
    voinames_numbered_dict = get_voinames_number(tp_plan_obj.voinames)

    if 'com' and 'tumor' in isocenter_method.lower():
        # Get a mask with only tumor data (1) and no tumor data(0)
        tumor_mask = (voinames_numbered_dict.get('tumor') == tp_plan_obj.voi)

        # Get center of mass location of tumor for isocenter
        isocenter_xy = calculate_centre_of_mass_voi(voi_mask=tumor_mask)

    elif 'com' and 'body' in isocenter_method.lower():
        # Get body mask
        body_mask = get_body_mask(tp_plan_obj=tp_plan_obj)

        # Get center of mass location of body for isocenter
        isocenter_xy = calculate_centre_of_mass_voi(voi_mask=body_mask)
    else:
        ValueError('An isocenter method is required to perform the raddepth calculation')

    print(f'Got isocenter_xy location: {isocenter_xy} for isocenter method: {isocenter_method}')
    return isocenter_xy


def calculate_centre_of_mass_voi(voi_mask:np.ndarray)->Dict[str,float]:
    """Calculates the centre of mass of a tumor given a mask and returns x,y coordinates in a dictionary form"""
    cy, cx = ndimage.center_of_mass(voi_mask)
    return {'x':cx,'y':cy}


def get_relative_attenuation_coeff(ct_scan:np.ndarray) ->np.ndarray:
    """ Perform piece-wise linear conversion from Hounsfield numbers to relative attenuation coeficcients
    For formula, check problemset 3. Use boolean indexing in numpy to select values of interest"""
    # Set any values below -1000 to -1000 (attenuation coefficient of air)
    ct_scan_clipped = np.clip(ct_scan, -1000, ct_scan.max())

    print(f"Check middle column for hounsfield units before conversion to attenuation coefficients {ct_scan_clipped[:, ct_scan_clipped.shape[1] // 2]}")
    # Initialize atten_coeff_matrix variable
    atten_coeff_matrix = ct_scan_clipped

    # Get mask for -1000 (include) and 0 (exclude) hounsfield units
    atten_below_water_mask = (atten_coeff_matrix >= -1000) & (atten_coeff_matrix < 0)

    # Get mask for above 0 (include) hounsfield units
    atten_above_water_mask = (atten_coeff_matrix >= 0)

    # Convert hounsfield units between -1000 (include) and 0 (exclude) to relative attenuation
    atten_coeff_matrix = np.where(atten_below_water_mask,
                                  (atten_coeff_matrix + 1000) / 1000,
                                  atten_coeff_matrix)

    # Convert hounsfield units above 0 (include) to relative attenuation
    atten_coeff_matrix = np.where(atten_above_water_mask,
                                  1 + (0.5 * atten_coeff_matrix) / 1000,
                                  atten_coeff_matrix)

    print(f"Check middle column after conversion to attenuation coefficients matrix {atten_coeff_matrix[:, atten_coeff_matrix.shape[1] // 2]}")
    return atten_coeff_matrix


def get_raddepth_from_atten_coeffs(atten_coeff_matrix:np.ndarray, isocenter:Dict[str,float], angle:Union[int,float], step_size=0.3, pixel_spacing=1)->np.ndarray:
    """Generate a radiological depth matrix from an attenuation coefficient matrix,
     an isocenter and an angle (from the y axis, clockwise)"""
    # List to track the first cast ray path (debugging)
    first_ray_path_x = []
    first_ray_path_y = []


    # Calculate ray direction vector from angle with trigonometric rotation formula (rays point toward isocenter)
    # formula for angle measured clockwise from +y-axis => dx = -sin(θ), dy = -cos(θ)
    dx = -np.sin(np.radians(angle))  # x-component of ray vector (right direction is positive, as x increases)
    dy = -np.cos(np.radians(angle))  # y-component of ray vector (upward direction is positive, as y increases)

    # Get array edge coordinates
    rows, cols = atten_coeff_matrix.shape
    outer_rim_mask = np.ones((rows, cols), dtype=bool)
    outer_rim_mask[1:-1, 1:-1] = False

    # Get all outer rim coordinates
    rim_coords = np.where(outer_rim_mask)
    rim_y, rim_x = rim_coords[0], rim_coords[1]

    # Extract isocenter coordinates from dictionary
    isocenter_x = isocenter['x']
    isocenter_y = isocenter['y']

    # Check if ray direction points toward isocenter from rim
    # For this we use the dot product test (formula)
    pointing_inward = ((isocenter_x - rim_x) * dx + (isocenter_y - rim_y) * dy) > 0

    # Get starting points where rays enter the image
    valid_starts_y = rim_coords[0][pointing_inward]
    valid_starts_x = rim_coords[1][pointing_inward]

    print(f"Found the x and y image edge starting positions for the rays:{[{'x':x,'y': y} for x,y in zip(valid_starts_x, valid_starts_y)]}")

    # Initialize radiological depth matrix
    raddepth_matrix = np.zeros_like(atten_coeff_matrix, dtype=float)
    visited = np.zeros_like(atten_coeff_matrix, dtype=bool)

    # Define a distance travelled per step, this might be relevant when pixel/voxel size is non-isotropic
    distance_per_step = pixel_spacing * step_size

    # Add the start x and y to the first ray debugging path
    first_ray_path_x.append(valid_starts_x[0])
    first_ray_path_y.append(valid_starts_y[0])

    # For each starting point, trace ray and calculate cumulative attenuation
    for start_y, start_x in zip(valid_starts_y, valid_starts_x):
        # Current position along the ray
        curr_x, curr_y = float(start_x), float(start_y)
        cumulative_depth = 0.0

        # Trace ray until it exits the array
        while 0 <= int(round(curr_y)) < rows and 0 <= int(round(curr_x)) < cols:
            # Get discrete pixel coordinates
            pixel_y, pixel_x = int(round(curr_y)), int(round(curr_x))

            # Add attenuation coefficient to cumulative depth
            cumulative_depth += atten_coeff_matrix[pixel_y, pixel_x] * distance_per_step

            if not visited[pixel_y,pixel_x]:
                # Store cumulative depth at this pixel
                raddepth_matrix[pixel_y, pixel_x] = cumulative_depth
                visited[pixel_y,pixel_x] =True
            else:
                pass

            # Move along ray direction (step size determines sampling resolution)
            curr_x += dx * step_size
            curr_y += dy * step_size

            # Follow the first beam for debugging
            if start_x == valid_starts_x[0] and start_y == valid_starts_y[0]:
                first_ray_path_x.append(curr_x), first_ray_path_y.append(curr_y)

    print(f"First ray x and y coords: {[{'x':x,'y': y} for x,y in zip(first_ray_path_x, first_ray_path_y)]}")
    print("Finished casting rays.")

    return raddepth_matrix


def calculate_raddepth(angle: Union[int|float], ct_scan:np.ndarray, isocenter:Dict[str,float]) -> np.ndarray:
    """Calculate the radiological depth from ct scan Hounsfield units for a given angle
    from the y-axis (clockwise) for a given isocentre as rotation point."""

    # Get relative attenuation coefficients
    atten_coeff_matrix = get_relative_attenuation_coeff(ct_scan=ct_scan)

    # Perform raytracing for angle and isocenter on relative_attenuation_coefficient_matrix
    raddepth_matrix = get_raddepth_from_atten_coeffs(atten_coeff_matrix=atten_coeff_matrix,
                                                     isocenter=isocenter, angle=angle)

    return raddepth_matrix


def raddepth_on_ct(tp_plan_path:Path, angle:Union[float|int], isocenter_method='COM_tumor', show_outside_body=False,
                   show_image:bool = True, show_isocenter:bool = False,
                   show_contours:bool = False, voinames_colors_visualization:Tuple[str,str] = None):
    """Plot the radiological depth of a ct scan on top of a ct scan as a colormap"""
    print("Calculating raddepth and showing the matrix on a ct")
    # Get CT data.
    tp_plan_obj, ct_img = visualize_tp_plan_data(tp_plan_path=tp_plan_path,
                           show_plot=False)

    # Get isocenter_xy coordinates using chosen method.
    # Currently supports methods with 'com' and 'tumor' or 'com' and 'body'
    isocenter_xy = get_isocenter(tp_plan_obj=tp_plan_obj,isocenter_method=isocenter_method)

    # Calculate the raddepth for a given angle on the CT. Pass isocenter, for calculation optimizations
    raddepth_matrix = calculate_raddepth(angle=angle,ct_scan=tp_plan_obj.ct,isocenter=isocenter_xy)

    # Keep raddepth data outside body or remove outside body data by setting to 0
    if not show_outside_body:
        body_mask = get_body_mask(tp_plan_obj=tp_plan_obj)
        outside_body = (body_mask == 0)
        raddepth_matrix = np.where(outside_body, 0, raddepth_matrix)

    # Plot the data
    raddepth_img = plt.imshow(raddepth_matrix, alpha=0.3,origin='lower', cmap='jet')
    plt.plot(isocenter_xy.get('x'), isocenter_xy.get('y'), 'r*', markersize=5, label='CoM - Isocenter')  # Red star
    plt.legend()
    plt.colorbar(raddepth_img, label=f'rad depth Zrad, angle: {angle} degs')
    plt.show()

if __name__ == '__main__':
    # #=====================================Week1===============================
    # # CT with contours visualization
    # visualize_tp_plan_data(
    #     tp_plan_path=Path(str(project_root_provider()) + r'.\utils\data\patientdata.mat'),
    #     voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','blue')],
    #     show_plot=True)

    # #=====================================Week2===============================
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

    # #=====================================Week3===============================

    # raddepth_on_ct(
    # tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     angle=60,
    #     isocenter_method='com_body'
    # )

    raddepth_on_ct(
    tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
        angle=100,
        isocenter_method='com_tumor',
        show_outside_body=True
    )

    raddepth_on_ct(
    tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
        angle=200,
        isocenter_method='com_tumor',
        show_outside_body=False
    )

    raddepth_on_ct(
    tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
        angle=300,
        isocenter_method='com_tumor',
        show_outside_body=False
    )