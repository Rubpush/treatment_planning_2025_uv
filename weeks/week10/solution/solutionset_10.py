"""
File: solutionset_10.py
Author: Ruben Bosschaert
Creation date: 20 Sept 2025
Description: This script provides the solution to problemset_10,

Simply run this file script and the name == main function at the end of the script should generate the relevant outputs. """

from typing import Union, List, Dict, Tuple, Any
from pathlib import Path

import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from numpy.distutils.system_info import flame_info
from scipy import ndimage
import math

from scipy.ndimage import gaussian_filter1d
from scipy.optimize import LinearConstraint

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

class Beamletdose:
    filepath: str = None
    dose: np.ndarray = None
    voxelsize: float = None
    x: np.ndarray = None
    z: np.ndarray = None
    beamletsize: np.ndarray = None
    centralaxis_x: float = None
    """
    Treatment beam container that supports both predefined and dynamic attributes.

    Predefined attributes:
        filepath, dose, voxelsize, x, z, beamletsize, centralaxis_x

    Additional attributes can be added dynamically via the constructor or from_dict().
    """

    def __init__(self, **kwargs):
        """Initialize the container with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        """String representation of the TPlan."""
        items = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"Beamletdose({', '.join(items)})"

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


def _load_mat_file(file_path: Path | str, data_type: str) -> dict:
    """Helper function to load and validate mat files"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Filepath to load {data_type} data does not exist: {path}")

    print(f"Loading {data_type} data from path: {path}")
    data_dict = mat_loader.loadmat(str(path))
    print(f"Success! Dictionary contents: {data_dict}")
    return data_dict


def load_tp_plan_data(tp_plan_path: Union[Path | str]) -> TPlan:
    """Loads data with the mat_loader and stores the data in a returned TPlan object"""
    # Get tp_plan_dict
    tp_plan_dict = _load_mat_file(tp_plan_path, "tp plan")

    # add filepath to the dictionary
    tp_plan_dict['TPlan']['filepath'] = tp_plan_path

    # load dictonary into TPlan object/data_model
    tp_plan_obj = TPlan.from_dict(tp_plan_dict['TPlan'])
    print(f"Loaded the dictionary data into a TPlan object: {tp_plan_obj}")

    return tp_plan_obj

def load_dose_data(dose_path: Union[Path | str], tplan:TPlan) -> TPlan:
    """Loads data with the mat_loader and stores the data in a returned TPlan object"""
    # Get dose_grid_dict
    dose_grid_dict = _load_mat_file(dose_path, "dose")

    # add dose data to TPlan object
    tplan.__setattr__('dosegrid',dose_grid_dict['dose'])

    return tplan

def load_beamletdose_data(beamletdose_path: Union[Path | str]) -> Beamletdose:
    """Loads data with the mat_loader and stores the data in a returned TPlan object"""
    # get beamletdose dict
    beamletdose_dict = _load_mat_file(beamletdose_path, "beamletdose")

    # add filepath to the dictionary
    beamletdose_dict['beamletdose']['filepath'] = beamletdose_path

    # load dictonary into TPlan object/data_model
    beamletdose_obj = Beamletdose.from_dict(beamletdose_dict['beamletdose'])
    print(f"Loaded the dictionary data into a beamletdose object: {beamletdose_obj}")

    return beamletdose_obj

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
                           show_plot:bool =True, plot_title: str = 'CT image plot'):
    """Visualize treatment planning data showing CT image with overlaid VOI contours.

    Args:
        tp_plan_path (Union[Path, str]): Path to treatment plan data file (.mat format).
        voinames_colors_visualization (List[Tuple[str, str]]): List of (voi_name, color) tuples.
            VOI names must match those in the data file. Colors should be valid matplotlib specs.
            If none, no contours will be plotted
        show_plot (bool): argument to show plot or not, if not shown, returns the plot, else returns none
        plot_title: The title the plot will have

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



    # Plot the contour data if
    if voinames_colors_visualization:
        voi_masks = {}
        for voiname, color in voinames_colors_visualization:
            voi_mask = plot_voi_contour(tp_plan_obj.voi,voinames_numbered_dict,voiname,color)
            voi_masks[f'{voiname}'] = voi_mask

    #Add legend
    plt.legend(fontsize=8)

    if show_plot:
        # Plot the ct imaging data
        ct_img = plt.imshow(tp_plan_obj.ct, cmap='grey', origin='lower')

        plt.title(plot_title)

        plt.show()
        return None, ct_img
    else:
        return tp_plan_obj, None

# #=====================================Week1===============================

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

            # Smooth the data by taking a weighted average of a region for better contours (although default 0.5 is low smoothing)
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


# #=====================================Week3===============================

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

def get_ray_direction_vector_from_angle(angle:float) -> (float,float):
    ''' Calculate ray direction vector from angle with trigonometric rotation formula (rays point toward isocenter)
    formula for angle measured clockwise from +y-axis => dx = -sin(θ), dy = -cos(θ)

    Returns dy, dx'''
    dx = -np.sin(np.radians(angle))  # x-component of ray vector (right direction is positive, as x increases)
    dy = -np.cos(np.radians(angle))  # y-component of ray vector (upward direction is positive, as y increases)
    return dy, dx

def get_ray_starting_positions(matrix_to_trace:np.ndarray, dx:float,dy:float, isocenter:Dict[str,float])-> (List[float],List[float]):
    """Gets ray starting positions to start raytracing in the direction of the angle.

    Returns valid_starts_y, valid_starts_x"""
    # Get array edge coordinates
    rows, cols = matrix_to_trace.shape
    outer_rim_mask = np.ones((rows, cols), dtype=bool)
    outer_rim_mask[1:-1, 1:-1] = False

    # Get all outer rim coordinates
    rim_coords = np.where(outer_rim_mask)
    rim_y, rim_x = rim_coords[0], rim_coords[1]

    # Extract isocenter coordinates from dictionary
    # UPDATE: Do not use isocenter anymore in this function and this pass 0 until function sig is updated
    isocenter_x = isocenter.get('x')
    isocenter_y = isocenter.get('y')

    # Check if ray direction points toward isocenter (or image center) from rim
    # For this we use the dot product test (v₁ · v₂ = v₁ₓ × v₂ₓ + v₁ᵧ × v₂ᵧ),
    # where v₁ = vector from rim to isocenter, and v₂ = ray direction
    # if positive, the x and or y components point in same direction,
    pointing_inward = ((isocenter_x - rim_x) * dx + (isocenter_y - rim_y) * dy) > 0
    # pointing_outward = ((isocenter_x - rim_x) * dx + (isocenter_y - rim_y) * dy) < 0

    # Get starting points where rays enter the image
    valid_starts_y = rim_coords[0][pointing_inward]
    valid_starts_x = rim_coords[1][pointing_inward]

    print(f"Found the x and y image edge starting positions for the rays:{[{'x':x,'y': y} for x,y in zip(valid_starts_x, valid_starts_y)]}")

    return valid_starts_y, valid_starts_x


def get_raddepth_from_atten_coeffs(atten_coeff_matrix:np.ndarray, isocenter:Dict[str,float], angle:Union[int,float], step_size=0.3, pixel_size:float=1)->np.ndarray:
    """Generate a radiological depth matrix from an attenuation coefficient matrix,
     an isocenter and an angle (from the y axis, clockwise)"""
    # List to track the first cast ray path (debugging)
    first_ray_path_x = []
    first_ray_path_y = []

    # Calculate ray direction vector from angle
    dy , dx = get_ray_direction_vector_from_angle(angle=angle)

    # Get ray x and y start positions
    rows, cols = atten_coeff_matrix.shape
    valid_starts_y, valid_starts_x = get_ray_starting_positions(matrix_to_trace=atten_coeff_matrix,dx=dx,dy=dy,
                                                                isocenter=isocenter)

    # Initialize radiological depth matrix
    raddepth_matrix = np.zeros_like(atten_coeff_matrix, dtype=float)
    visited = np.zeros_like(atten_coeff_matrix, dtype=bool)

    # Define a distance travelled per step, this might be relevant when pixel/voxel size is non-isotropic
    distance_per_step = pixel_size * step_size

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


def calculate_raddepth(angle: Union[int|float], ct_scan:np.ndarray, isocenter:Dict[str,float],pixel_size:float, step_size:float=0.5) -> np.ndarray:
    """Calculate the radiological depth from ct scan Hounsfield units for a given angle
    from the y-axis (clockwise) for a given isocentre as rotation point."""

    # Get relative attenuation coefficients
    atten_coeff_matrix = get_relative_attenuation_coeff(ct_scan=ct_scan)

    # Perform raytracing for angle and isocenter on relative_attenuation_coefficient_matrix
    raddepth_matrix = get_raddepth_from_atten_coeffs(atten_coeff_matrix=atten_coeff_matrix,
                                                     isocenter=isocenter, step_size=step_size,pixel_size=pixel_size,
                                                     angle=angle)

    return raddepth_matrix


def raddepth_on_ct(tp_plan_path:Path, angle:Union[float|int], isocenter_method='COM_tumor', step_size=0.2, show_outside_body=False,
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
    print(f"Input step_size ratio {step_size}, on the voxel grid {tp_plan_obj.voxelsize} mm relates to step_size {tp_plan_obj.voxelsize*step_size} mm")
    raddepth_matrix = calculate_raddepth(angle=angle,ct_scan=tp_plan_obj.ct,isocenter=isocenter_xy,pixel_size=tp_plan_obj.voxelsize, step_size=step_size)

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

 # #=====================================Week4===============================



def visualize_beamletdose_data(beamletdose_path: Union[Path | str],show_plot:bool =True,
        plot_title: str = 'Beamlet image plot'):
    """Visualize treatment planning data showing CT image with overlaid VOI contours.

    Args:
        beamletdose_path (Union[Path, str]): Path to treatment plan data file (.mat format).
        show_plot (bool): argument to show plot or not, if not shown, returns the plot, else returns none
        plot_title: The title the plot will have

    Returns:
        None: Displays plot with plt.show().

    Raises:
        ValueError: If inputs are None/empty.

    Example:
        visualize_beamletdose_data(Path('data.mat'),True)
    """
    if not beamletdose_path:
        raise ValueError(f"Missing input arguments for beamletdose_path")

    # Get TPlan object
    beamletdose_obj = load_beamletdose_data(beamletdose_path)



    if show_plot:
        # Plot the ct imaging data
        beamletdose_img = plt.imshow(beamletdose_obj.dose,  cmap='jet', origin='lower')
        plt.title(plot_title)

        plt.show()
        return None, beamletdose_img
    else:
        return beamletdose_obj, None


def interpolate_to_grid(from_img_data: np.ndarray,from_voxelsize:float = 0.5, to_img_data:np.ndarray = None, to_voxelsize:float =2.5)-> np.ndarray:
    """interpolate input image from voxelsize to voxelsize."""
    # Get image shapes, also in seperate y and x variables
    from_img_shape, to_img_shape = np.shape(from_img_data), np.shape(to_img_data)
    print(f'interpolating image data with shape: {from_img_shape} and voxelsize {from_voxelsize} to voxelsize {to_voxelsize}')
    # y_from_img_shape, x_from_img_shape = from_img_shape
    # y_to_img_shape, x_to_img_shape = to_img_shape

    # Get ratio of input to output for each axis
    out_to_in_ratio = from_voxelsize/to_voxelsize

    # Use interpolate zoom from scipy for b spline interpolation
    data_interpolated = ndimage.interpolation.zoom(input=from_img_data,
                                                   zoom=out_to_in_ratio)

    print(f"Interpolated data to shape:{np.shape(data_interpolated)}")
    return data_interpolated


def rotate_and_align_beam(beam_img, ref_img, angle, isocenter, latpos=0, longpos=0):
    """
    Rotate beam_img around its center and align the beam start voxel to edge,
    ensuring the beam passes through (isocenter + latpos/longpos).

    Algorithm overview:
    1. Find the beam start point (highest intensity) in the original beam image
    2. Rotate the beam image by the specified angle
    3. Track where the beam start moved after rotation
    4. Calculate which edge the beam should enter from
    5. Position the rotated beam so it enters at the edge and passes through target

    Parameters:
    - beam_img: Input 2D or 3D image with beam (high-intensity start)
    - ref_img: Reference image or shape
    - angle: Rotation angle in degrees (clockwise)
    - isocenter: {'y': int, 'x': int} target coordinates in ref_img
    - latpos: lateral shift (perpendicular to beam axis)
    - longpos: longitudinal shift (along beam axis)
    """
    print('Rotating beam and aligning to reference image')

    # Beam starts from x=0 and moves along positive x-axis
    beam_start_x = 0

    # beam image center coords
    h, w = beam_img.shape
    center_y, center_x = h / 2, w / 2

    # Offset from center to beam start
    offset_y = 0
    offset_x = beam_start_x - center_x

    # Rotate beam_img around its center
    rotated = ndimage.rotate(beam_img, angle, reshape=True, order=1, mode='constant', cval=0)

    rot_h, rot_w = rotated.shape[:2]
    rot_center_y, rot_center_x = rot_h / 2, rot_w / 2

    # Transform offset vector to track beam start position in rotated image.
    # Image rotates by +angle, but output coordinates stay axis-aligned,
    # so we apply inverse rotation (-angle) to find new beam start position.
    theta = np.deg2rad(angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    rot_offset_x = cos_theta * offset_x + sin_theta * offset_y
    rot_offset_y = -sin_theta * offset_x + cos_theta * offset_y

    # Beam direction from angle (positive angle is clockwise)
    beam_angle_rad = np.deg2rad(angle)
    # When plotted with origin lower, dx and dy align with increasing x and y axis.
    # Minus x is towards left, minus y is towards bottom
    dx = np.cos(beam_angle_rad)  # x-component
    dy = -np.sin(beam_angle_rad)  # y-component

    # Perpendicular direction to beam (lateral direction)
    # Rotate beam direction by 90° counterclockwise: (dx, dy) -> (-dy, dx)
    lateral_dx = -dy
    lateral_dy = dx

    # Target point that beam should pass through
    # Apply offsets in beam-relative coordinates:
    # - longpos: along the beam direction
    # - latpos: perpendicular to the beam direction
    grid_shape = ref_img.shape[:2]
    target_x = isocenter['x'] + longpos * dx + latpos * lateral_dx
    target_y = isocenter['y'] + longpos * dy + latpos * lateral_dy

    # Find which edge the beam enters from and calculate exact entry coordinates
    # Using parametric line equation: point = start + t * direction
    # We solve for the edge point where the line through target intersects the edge
    if abs(dx) > abs(dy):
        # Beam is mostly horizontal
        if dx > 0:
            # Beam travels rightward → enters from left edge (x=0)
            edge_x = 0
            # Solve for y: edge_point + t*(dy,dx) = target
            # From x-component: 0 + t*dx = target_x → t = target_x/dx
            # Substitute into y-component: edge_y + t*dy = target_y
            edge_y = target_y - (target_x / dx) * dy if abs(dx) > 1e-10 else target_y
        else:
            # Beam travels leftward → enters from right edge
            edge_x = grid_shape[1] - 1
            edge_y = target_y - ((target_x - edge_x) / dx) * dy if abs(dx) > 1e-10 else target_y
    else:
        # Beam is mostly vertical
        if dy > 0:
            # Beam travels downward → enters from top edge (y=0)
            edge_y = 0
            edge_x = target_x - (target_y / dy) * dx if abs(dy) > 1e-10 else target_x
        else:
            # Beam travels upward → enters from bottom edge
            edge_y = grid_shape[0] - 1
            edge_x = target_x - ((target_y - edge_y) / dy) * dx if abs(dy) > 1e-10 else target_x

    # Calculate where to place the rotated beam image
    # We need the beam start point (which is at rot_center + rot_offset in the rotated image)
    # to align with the edge entry point we just calculated
    beam_start_in_rotated_y = rot_center_y + rot_offset_y
    beam_start_in_rotated_x = rot_center_x + rot_offset_x

    # These offsets position the rotated image so beam start aligns with edge entry
    top = int(round(edge_y - beam_start_in_rotated_y))
    left = int(round(edge_x - beam_start_in_rotated_x))

    # Prepare output image
    if rotated.ndim == 3:
        result = np.zeros((grid_shape[0], grid_shape[1], rotated.shape[2]), dtype=rotated.dtype)
    else:
        result = np.zeros(grid_shape, dtype=rotated.dtype)

    # Calculate the overlapping region between rotated beam and reference image
    # If beam is placed at top=-10, left=20:
    # Beam's top 10 rows would be outside ref image (above top edge)
    # -> We skip those rows and only copy what's visible

    # Source region (what part of the rotated beam to copy)
    if top < 0:
        src_y_start = -top  # Skip rows that would be above ref image
    else:
        src_y_start = 0  # Start from beginning of rotated image

    if left < 0:
        src_x_start = -left  # Skip columns that would be left of ref image
    else:
        src_x_start = 0  # Start from beginning of rotated image

    # Don't copy beyond what fits in the reference image
    src_y_end = min(rot_h, grid_shape[0] - top)
    src_x_end = min(rot_w, grid_shape[1] - left)

    # Destination region (where to paste in the reference image)
    dst_y_start = max(0, top)  # Clamp to image bounds
    dst_x_start = max(0, left)  # Clamp to image bounds

    # End positions maintain the same size as source region
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    # Only copy the overlapping region of the rotated beam image on the ct_image
    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = rotated[src_y_start:src_y_end, src_x_start:src_x_end]

    return result

def stretch_and_compress_beam_raddepth(beam:np.ndarray, raddepth_matrix:np.ndarray, angle, isocenter,voxelsize,stepsize=0.2,
                                       beam_start_threshold=0.001)-> np.ndarray:
    """Map a radiation beam onto patient anatomy, adjusting for tissue density variations.

    Casts rays through the patient's radiological depth map and samples beam intensity
    values at positions determined by accumulated radiological depth. This accounts for
    tissue density: rays travel further in low-density tissue (e.g., lung) and compress
    in high-density tissue (e.g., bone).

    PS: Check Siddon and Brensemham algorithm for inspiration

    Args:
        beam: 2D intensity profile of the radiation beam
        raddepth_matrix: Radiological depth map derived from CT (accounts for tissue density)
        angle: Beam angle in degrees
        isocenter: Point where beam is aimed (y, x)
        voxelsize: Physical size of each voxel in mm
        stepsize: Ray marching step size (fraction of voxel)

    Returns:
        Masked 2D array of dose distribution in patient space
    """
    print('Stretch and compress beam according to raddepth')

    # Calculate ray direction vector from angle
    dy , dx = get_ray_direction_vector_from_angle(angle=angle)

    # Get ray x and y start positions
    valid_starts_y, valid_starts_x = get_ray_starting_positions(matrix_to_trace=beam,dx=dx,dy=dy,isocenter=isocenter)

    # Initialize dose in pat matrix
    dose_in_pat_matrix = np.zeros_like(raddepth_matrix, dtype=float)
    visited = np.zeros_like(raddepth_matrix, dtype=bool)

    distance_per_step = voxelsize * stepsize
    rows, cols = raddepth_matrix.shape

    # Calculate beam intensity threshold (1% of max)
    beam_max = np.max(beam)
    beam_threshold = beam_start_threshold * beam_max  # 1% threshold for starting ray marching


    # Track ray path containing max intensity beam voxels for debugging and visualization
    max_ray_path_x , max_ray_path_y, max_beam_path_values = [], [], []
    highest_ray_beam_val = 0  # Track the highest beam value found in a ray


    # For each starting point, trace ray and through patient calculate if beam needs to be stretched or compressed
    for start_y, start_x in zip(valid_starts_y, valid_starts_x):

        # Find how much distance to skip along the beam ray before reaching threshold
        beam_skip_distance = 0
        beam_y, beam_x = float(start_y), float(start_x)

        # March along the ray in BEAM space to find where intensity exceeds threshold
        while 0 <= int(round(beam_y)) < rows and 0 <= int(round(beam_x)) < cols:
            pixel_y, pixel_x = int(round(beam_y)), int(round(beam_x))

            # Check if we've reached the threshold intensity
            if beam[pixel_y, pixel_x] >= beam_threshold:
                # Found where the beam becomes significant
                # Calculate the distance we've traveled in beam space
                beam_skip_distance = np.sqrt((beam_y - start_y) ** 2 + (beam_x - start_x) ** 2)
                break

            # Move to next position along ray in beam space
            beam_y += dy
            beam_x += dx

        # If we never found a point above threshold, skip this ray entirely
        if beam_skip_distance == 0 and beam[int(round(start_y)), int(round(start_x))] < beam_threshold:
            continue

        curr_rad_x, curr_rad_y = float(start_x), float(start_y)
        ray_path_x, ray_path_y, beam_path_values = [], [],[]

        # March ray through radiological depth matrix
        # for each y and x coordinate, check raddepth value and paint with value corresponding to x and y of the beam(depth)
        while 0 <= int(round(curr_rad_y)) < rows and 0 <= int(round(curr_rad_x)) < cols:
            # Get discrete pixel coordinates
            pixel_y, pixel_x = int(round(curr_rad_y)), int(round(curr_rad_x))

            # Get ray raddepth and calculate beam depth - distance along beam based on accumulated radiological depth
            ray_raddepth = raddepth_matrix[pixel_y, pixel_x] * distance_per_step
            beam_depth = ray_raddepth + beam_skip_distance

            # Map back to beam coordinates based on accumulated depth
            beam_y = int(round(start_y + beam_depth * dy))
            beam_x = int(round(start_x + beam_depth * dx))

            # Store the ray x and y coordinates in the radiological depth space
            ray_path_x.append(curr_rad_x), ray_path_y.append(curr_rad_y)

            # Transfer beam intensity to patient voxel (if unvisited and within bounds)
            if 0 <= beam_y < rows and 0 <= beam_x < cols and not visited[pixel_y, pixel_x]:
                dose_in_pat_matrix[pixel_y, pixel_x] = beam[beam_y, beam_x]
                visited[pixel_y, pixel_x] = True

                 # Store mapped beam values
                beam_path_values.append(beam[beam_y, beam_x])

            # Move to next position in radiological depth space
            curr_rad_x += dx
            curr_rad_y += dy

        # Check the ray if it contains the maximum value intensity in the beam image, if so save for debug
        if beam_path_values:
            # Use maximum value encountered
            max_ray_beam_val = np.max(beam_path_values)

            # If this ray has the highest beam val so far, save it
            if max_ray_beam_val > highest_ray_beam_val:
                highest_ray_beam_val = max_ray_beam_val
                max_ray_path_x = ray_path_x
                max_ray_path_y = ray_path_y
                max_beam_path_values = beam_path_values

    print(f"Highest intensity ray x and y coords: {[{'x': x, 'y': y} for x, y in zip(max_ray_path_x, max_ray_path_y)]}")
    print(f"highest intensity ray beam values along path:{max_beam_path_values}")
    print("Finished casting rays.")

    return dose_in_pat_matrix


def calculate_pencil_beam_dose(beamlet_dose:np.ndarray, angle:float, latpos:Dict[str,float]
                               ,isocenter, raddepth: np.ndarray,voxelsize:float)->Dict[str,Any]:

    # Rotate and align the beam array with the raddepth array for the isocenter and latpos
    # Angle +90 to align with raddepth method for ndarray rotate
    rotated_padded_beam = rotate_and_align_beam(beamlet_dose, raddepth, angle=(angle + 90), isocenter=isocenter, latpos=latpos)

    # Stretch and compress the beam according to the raddepth
    beam_corrected_for_raddepth = stretch_and_compress_beam_raddepth(beam=rotated_padded_beam,raddepth_matrix=raddepth,
                                                                     angle=angle,voxelsize=voxelsize, isocenter=isocenter,
                                                                     stepsize=0.2)

    return {'angle':angle,'latpos':latpos,'dose':beam_corrected_for_raddepth}


def show_pencil_beam_dose_on_ct(tp_plan_path:Path, beamletdose_path:Path, angle:float, latpos:Dict[str,float],
                                voinames_colors_visualization: List[Tuple[str, str]] = None,
):
    show_outside_body = False

    # Get CT data.
    tp_plan_obj, _ = visualize_tp_plan_data(tp_plan_path=tp_plan_path,
                           show_plot=False, voinames_colors_visualization=voinames_colors_visualization)

    # Get beamlet data
    beamletdose_obj, beamlet_img = visualize_beamletdose_data(beamletdose_path=beamletdose_path,
                           show_plot=False)

    # Get isocenter (Center of Mass tumor)
    isocenter = get_isocenter(tp_plan_obj, 'COM_tumor')

    # Get Raddepth matrix from ct for angle, boxelsize and stepsize
    raddepth_ct = calculate_raddepth(ct_scan=tp_plan_obj.ct,isocenter=isocenter,
                       pixel_size=tp_plan_obj.voxelsize,step_size=0.2, angle=angle)

    # Interpolate beamlet to ct grid
    beamlet_dose_on_ct_grid = interpolate_to_grid(from_img_data=beamletdose_obj.dose, from_voxelsize=beamletdose_obj.voxelsize,
                                            to_voxelsize=tp_plan_obj.voxelsize)
    # Get pencil beam dose
    pb = calculate_pencil_beam_dose(beamlet_dose_on_ct_grid, angle,
                                    latpos.get('x'), isocenter, raddepth_ct,
                                    voxelsize=tp_plan_obj.voxelsize)

    # Remove values outside pat
    if not show_outside_body:
        body_mask = get_body_mask(tp_plan_obj=tp_plan_obj)
        outside_body = (body_mask == 0)
        beam_dose = np.where(outside_body, 0, pb.get('dose'))

    # Remove values below chosen percentage
    remove_below_percentage = 1

    # Remove/mask dose values below cutoff
    beam_dose = np.ma.masked_where(beam_dose < ((remove_below_percentage / 100) * beam_dose.max()), beam_dose)

    ct_img = plt.imshow(tp_plan_obj.ct, cmap='grey', origin='lower')
    beam_ax = plt.imshow(beam_dose, cmap='jet', origin='lower', alpha=0.5)

    # Get the current axes object
    ax = plt.gca()

    # Convert axis labels from voxel indices to mm
    # Get current tick locations from the axes (not the image)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Assuming voxelsize is a scalar or [x_size, y_size, z_size]
    voxelsize_x = tp_plan_obj.voxelsize
    voxelsize_y = tp_plan_obj.voxelsize

    # Set new tick labels in mm on the axes (not the image)
    ax.set_xticklabels([f'{tick * voxelsize_x:.1f}' for tick in xticks])
    ax.set_yticklabels([f'{tick * voxelsize_y:.1f}' for tick in yticks])

    # Add axis labels to the axes (not the image)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')

    plt.colorbar(beam_ax, label=f'beamlet dose [Gy]')
    plt.title(f'Beamlet in patient, angle:{angle}, latpos: {latpos.get("x")} mm')
    plt.show()

 # #=====================================Week5===============================

def define_bethe_bloch_constants_variables() -> Dict:
    """Define all constants in SI units as specified in the problem set"""
    proton_rest_mass = 1.672631e-27  # kg
    electron_rest_mass = 9.1093897e-31  # kg
    vacuum_permittivity = 8.854187817e-12  # C^2/(J·m)
    charge_of_the_electron = 1.60217733e-19  # Coulombs
    speed_of_light = 2.99792458e8  # m/s
    avogadros_number = 6.0221367e23  # particles per mole (FIXED: was 20**23!)

    # Water properties
    ionization_potential_water = 75  # eV
    mass_density_water = 1000  # kg/m^3 (1 g/cm^3)
    molar_mass_water = 1.0e-3  # kg/mol for H2O
    electrons_per_molecule_water_ratio = (10/18)

    constants_variables = {
        'mp': proton_rest_mass,
        'me': electron_rest_mass,
        'epsilon0': vacuum_permittivity,
        'e': charge_of_the_electron,
        'c': speed_of_light,
        'Na': avogadros_number,
        'Mu_water': molar_mass_water,
        'I_water': ionization_potential_water,
        'Z_water': electrons_per_molecule_water_ratio,
        'rho_water': mass_density_water
    }

    return constants_variables


def get_electron_density(mass_density: float, molar_mass: float,
                         electrons_per_molecule: float, avogadros_number: float) -> float:
    """
    Calculate electron density Ne using formula.

    Ne = ρ * (NA/Mu) * Z

    Where:
    - ρ: mass density (kg/m³)
    - NA: Avogadro's number (molecules/mol)
    - Mu: molar mass (kg/mol)
    - Z: electrons per molecule

    Returns: electrons per m^3
    """
    # ρ/Mu = molar density (mol/m³)
    # (ρ/Mu) * NA = molecules/m³
    # (ρ/Mu) * NA * Z = electrons/m³
    electron_density = mass_density * (avogadros_number / molar_mass) * electrons_per_molecule
    return electron_density


def calculate_beta_squared_from_energy(energy_joules: float, mp: float, c: float) -> float:
    """
    Calculate β² from kinetic energy using equation (5):
    E = mp*c²/√(1-β²) - mp*c²

    Solving for β²:
    β² = E(E + 2mp*c²) / (E + mp*c²)²
    """
    if energy_joules <= 0:
        return 0.0

    mp_c2 = mp * c ** 2
    beta_squared = (energy_joules * (energy_joules + 2 * mp_c2)) / ((energy_joules + mp_c2) ** 2)

    # beta squared is expected to be within 0 and 1 due to dominant denominator. It cannot be negative (not phyisical)
    if beta_squared >= 1.0 or beta_squared <= 0:
        return 0.0

    return beta_squared


def bethe_bloch_stopping_power(energy_joules: float, const_vars: Dict, Ne: float, I_joules: float) -> float:
    """
    Calculate stopping power S(E) = -dE/dz using Bethe-Bloch equation (1):

    -dE/dz = (4πe⁴)/(me*c²) * (Ne/β²) * (1/(4πε₀))² * [ln(2me*c²*β²/(I(1-β²))) - β²]

    Returns: -dE/dz in J/m (negative value representing energy loss)
    """
    if energy_joules <= 0:
        return 0.0

    # Extract constants
    me = const_vars['me']
    mp = const_vars['mp']
    c = const_vars['c']
    e = const_vars['e']
    epsilon0 = const_vars['epsilon0']

    # Calculate β²
    beta_squared = calculate_beta_squared_from_energy(energy_joules, mp, c)

    if beta_squared >= 1.0 or beta_squared <= 0:
        return 0.0

    # Bethe-Bloch formula - equation (1)
    # Term 1: (1/(4πε₀))²
    term1 = (1.0 / (4.0 * np.pi * epsilon0)) ** 2

    # Term 2: (4πe⁴*Ne)/(me*c²*β²)
    me_c2 = me * c ** 2
    term2 = (4.0 * np.pi * e ** 4 * Ne) / (me_c2 * beta_squared)

    # Term 3: ln(2me*c²*β²/(I(1-β²))) - β²
    numerator = 2.0 * me_c2 * beta_squared
    denominator = I_joules * (1.0 - beta_squared)

    if denominator <= 0:
        return 0.0

    term3 = np.log(numerator / denominator) - beta_squared

    # Complete Bethe-Bloch
    stopping_power = -term1 * term2 * term3

    return stopping_power


def solve_bethe_bloch_csda(initial_energy_mev: float,
                           step_size_mm: float = 0.0001,
                           max_depth_mm: float = 500) -> Tuple:
    """
    Solve Bethe-Bloch equation using Euler method (equation 3):
    [E(z + Δz) - E(z)] / Δz = S(E(z))

    Returns depth-dose curve in CSDA (Continuous Slowing Down Approximation).
    """
    # Get constants
    const_vars = define_bethe_bloch_constants_variables()

    # Calculate electron density for water
    Ne = get_electron_density(
        mass_density=const_vars['rho_water'],
        molar_mass=const_vars['Mu_water'],
        electrons_per_molecule=const_vars['Z_water'],
        avogadros_number=const_vars['Na']
    )

    print(f"Electron density: Ne = {Ne:.4e} electrons/m³")

    # Convert ionization potential to Joules
    I_joules = const_vars['I_water'] * const_vars['e']

    # Convert units
    step_size_m = step_size_mm * 1e-3  # mm to m
    energy_joules = initial_energy_mev * 1e6 * const_vars['e']  # MeV to Joules

    # Initialize
    z_list = [0.0]
    E_list = [initial_energy_mev]
    dose_list = []

    z_current = 0.0
    E_current = energy_joules

    # Euler integration
    max_iterations = int(max_depth_mm / step_size_mm) + 1
    for iteration in range(max_iterations):
        # Calculate stopping power at current energy
        S_current = bethe_bloch_stopping_power(E_current, const_vars, Ne, I_joules)

        # Convert stopping power to MeV/mm for dose curve
        # S_current is in J/m, convert to MeV/mm
        dose_mev_per_mm = -S_current / (const_vars['e'] * 1e6) / 1e3
        dose_list.append(dose_mev_per_mm)

        # Update energy: E_new = E_old + S(E_old) * Δz
        E_new = E_current + S_current * step_size_m

        # Stop if energy drops to zero or becomes very small
        # Improvement: implement adaptive threshold
        if E_new <= 0.0000001 * energy_joules:
            E_new = 0
            z_current += step_size_m
            z_list.append(z_current * 1e3)
            E_list.append(0)
            # No dose at zero energy
            break

        # Update position
        z_current += step_size_m
        E_current = E_new

        # Store values
        z_list.append(z_current * 1e3)  # Convert to mm
        E_list.append(E_current / (const_vars['e'] * 1e6))  # Convert to MeV

    range_mm = z_list[-1]

    # Convert to numpy arrays (ensure all have same length)
    z_mm = np.array(z_list[:-1])  # Remove last element to match dose_list
    E_mev = np.array(E_list[:-1])
    dose_mev_per_mm = np.array(dose_list)

    return z_mm, E_mev, dose_mev_per_mm, range_mm


def apply_range_straggling(z_mm: np.ndarray, dose: np.ndarray, range_cm: float) -> np.ndarray:
    """
    Apply range straggling by convolving dose with Gaussian (equation 4):
    σ_R = 0.012 * R^0.935

    where R is the range in centimeters.
    """
    # Calculate standard deviation from equation (4)
    sigma_cm = 0.012 * (range_cm ** 0.935)
    sigma_mm = sigma_cm * 10  # Convert cm to mm

    # Convert sigma to array indices
    dz = z_mm[1] - z_mm[0] if len(z_mm) > 1 else 0.01
    sigma_indices = sigma_mm / dz

    # Apply Gaussian filter (convolution with Gaussian)
    dose_with_straggling = gaussian_filter1d(dose, sigma=sigma_indices, mode='constant')

    # # Normalize to preserve total energy deposition
    # if np.sum(dose_with_straggling) > 0:
    #     dose_with_straggling *= np.sum(dose) / np.sum(dose_with_straggling)

    return dose_with_straggling


def plot_proton_depth_dose_results(z, E, dose_csda, dose_straggling, range, initial_energy):
    """Plot the results"""
    fig, (ax) = plt.subplots(figsize=(12, 10))

    ax.plot(z, E, 'blue',  label='E (residual)', linewidth=2)
    ax.plot(z, dose_csda, 'red', linewidth=2, label='CSDA (without straggling)', alpha=0.7)
    ax.plot(z, dose_straggling, 'black', linewidth=2, label='With range straggling')
    ax.axvline(range, color='gray', linestyle='--', alpha=0.2)
    ax.set_xlabel('Depth z (cm)', fontsize=12)
    ax.set_ylabel('Proton residual energy (MeV), Energy Deposition -dE/dz (MeV/cm)', fontsize=12)
    ax.set_title(f'Bragg Curve with range: {range:.2f} cm', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(left=0, right=range*1.35)
    ax.set_ylim(bottom=0, top= np.max(E)*1.35)

    plt.tight_layout()
    return fig


def verify_range(initial_energy_mev: float, calculated_range_mm: float):
    """Verify the calculated range against NIST/PSTAR reference data"""
    # Reference data from NIST/PSTAR database
    reference_data = {
        10: 1.25,  # mm
        30: 8.96,  # mm
        50: 22,  # mm
        100: 77.93,  # mm
        200: 260,  # mm
        300: 518.7  # mm
    }

    print("\n" + "=" * 70)
    print("RANGE VERIFICATION")
    print("=" * 70)
    print(f"Initial Energy: {initial_energy_mev} MeV")
    print(f"Calculated Range: {calculated_range_mm:.2f} mm ({calculated_range_mm / 10:.2f} cm)")

    # Find closest reference energy
    closest_energy = min(reference_data.keys(), key=lambda x: abs(x - initial_energy_mev))
    if abs(closest_energy - initial_energy_mev) < 30:
        expected_range = reference_data[closest_energy]
        error_percent = abs(calculated_range_mm - expected_range) / expected_range * 100
        print(f"\nReference (NIST/PSTAR) for {closest_energy} MeV: ~{expected_range} mm")
        print(f"Difference: {error_percent:.1f}%")

        if error_percent < 1:
            print("Result is - EXCELLENT - within 1% of NIST data!")
        elif error_percent < 2.5:
            print("Result is - GOOD - within 2.5% of NIST data")
        elif error_percent < 5:
            print("Result is - OKAY - within 5% of NIST data")
        else:
            print("Result differs significantly from reference")
    print("=" * 70 + "\n")


def bethe_bloch_depth_dose_debug_table(dose_csda,dose_straggling, E, z, range_cm):
    # Find Bragg peak locations
    peak_idx_csda = np.argmax(dose_csda)
    peak_idx_strag = np.argmax(dose_straggling)


    print(f"Results Summary:")
    print(f"  Range: {range_cm*10:.2f} mm ({range_cm:.2f} cm)")
    print(f"  Number of integration steps: {len(z)}")
    print(f"  CSDA Bragg peak:")
    print(f"    - Position: {z[peak_idx_csda]:.2f} mm")
    print(f"    - Height: {dose_csda[peak_idx_csda]:.2f} MeV/mm")
    print(f"  With range straggling:")
    print(f"    - Peak position: {z[peak_idx_strag]:.2f} mm")
    print(f"    - Peak height: {dose_straggling[peak_idx_strag]:.2f} MeV/mm")
    print(f"  Range straggling σ: {0.012 * (range_cm ** 0.935) * 10:.2f} mm")

    # Show sample data points
    print(f"\n  Sample depth-dose data:")
    print(f"  {'Depth(mm)':<12} {'Energy(MeV)':<14} {'CSDA(MeV/mm)':<16} {'Straggling(MeV/mm)':<20}")
    print(f"  {'-' * 72}")

    # Sample at different depths
    sample_indices = [
        0,  # Start
        len(z) // 10,  # 10%
        len(z) // 4,  # 25%
        len(z) // 2,  # 50%
        3 * len(z) // 4,  # 75%
        peak_idx_csda,  # Peak
        min(len(z) - 1, peak_idx_csda + 5)  # Just past peak
    ]

    for i in sorted(set(sample_indices)):
        if i < len(z):
            marker = " ← BRAGG PEAK" if i == peak_idx_csda else ""
            print(f"  {z[i]:<12.2f} {E[i]:<14.2f} {dose_csda[i]:<16.3f} "
                  f"{dose_straggling[i]:<20.3f}{marker}")


def calculate_and_plot_residual_e_and_de_dz_with_and_without_range_straggling(
        energies_to_test:List[int]=[10, 50, 100, 300],
        debug_table:bool=False
):
    print("\n" + "=" * 70)
    print("BETHE-BLOCH DOSE-DEPTH CALCULATION")
    print("Proton therapy in water - CSDA with range straggling")
    print("=" * 70 + "\n")

    for initial_energy in energies_to_test:
        print(f"\n{'=' * 70}")
        print(f"Solving for {initial_energy} MeV proton in water")
        print(f"{'=' * 70}")

        # Solve Bethe-Bloch in CSDA (without range straggling)
        z, E, dose_csda, range_mm = solve_bethe_bloch_csda(
            initial_energy_mev=initial_energy,
            step_size_mm=0.001,
            max_depth_mm=500
        )

        # Apply range straggling (equation 4)
        range_cm = range_mm / 10
        dose_straggling = apply_range_straggling(z, dose_csda, range_cm)

        # Verify against NIST data
        verify_range(initial_energy, range_mm)

        if debug_table:
            bethe_bloch_depth_dose_debug_table(dose_csda,dose_straggling, E, z, range_cm)

        # Create and save plot
        # Convert from mm to cm
        dose_csda = dose_csda * 10
        dose_straggling = dose_straggling * 10
        z = z / 10

        fig = plot_proton_depth_dose_results(z, E, dose_csda, dose_straggling, range_cm, initial_energy)
        filename = f'proton_depth_dose_curve_{initial_energy}MeV.png'
        plt.savefig(f'{filename}', dpi=150, bbox_inches='tight')
        print(f"\n Plot saved: {filename}")
        plt.close()

    print(f"\n{'=' * 70}")
    print("Finished Bethe Bloch dose depth curve calculations!")
    print("=" * 70)


# #=====================================Week6===============================
def get_proton_dose_data(protondosesfile_path:Path) -> Dict[str,Any]:
    """Load proton beam dose data from file"""
    print(f'Loading proton dose data from: {protondosesfile_path}')

    # Use numpy to parse data from dose file
    text = numpy.loadtxt(protondosesfile_path)

    return {'proton_dose_data' : text,
            'proton_dose_data_depth_mm': text[0:len(text),0],
            'proton_dose_data_d0': text[0:len(text),1],
            'proton_dose_data_sigma': text[0:len(text),2],}


def calculate_proton_pencil_beam_dose(angle, init_energy, latpos, raddepth, tp_plan_obj,
                                      protondose_data=None) -> Dict:
    """
    Calculate proton pencil beam dose distribution using direct lookup (no interpolation).

    The dose is calculated using a Gaussian pencil beam model:
    D(x,y,z) = D0(z,E0) * [1/(2πσ²)] * exp[-(x²+y²)/(2σ²)]

    For 2D calculations, we set y=0 in the beam coordinate system.

    Parameters:
    -----------
    angle : float
        Beam angle in degrees using convention:
        0° = top to bottom, 90° = right to left, 180° = bottom to top, 270° = left to right
    init_energy : float
        Initial proton energy in MeV
    latpos : dict
        Lateral position of beam's central axis relative to isocenter {'x': float, 'y': float} in mm
    raddepth : numpy.ndarray
        Radiological depth matrix for the beam angle (in ct grid)
    tp_plan_obj : object
        Treatment plan object containing CT data and voxel size
    protondose_data : dict, optional
        Pre-loaded proton dose data (depth, D0, sigma)
    protondosesfolder_path : Path, optional
        Path to proton dose data folder (if protondose_data not provided)

    Returns:
    --------
    dict : Dictionary containing angle, energy, latpos, and dose distribution
    """

    # =====================================================================
    # Initialize dose calculation
    # =====================================================================
    # Extract depth-dependent beam parameters from lookup table
    depth_mm = protondose_data['proton_dose_data_depth_mm']  # Depth in water (mm)
    D0 = protondose_data['proton_dose_data_d0']  # Central axis dose
    sigma = protondose_data['proton_dose_data_sigma']  # Beam width (standard deviation)

    # Get CT dimensions
    ct_shape = raddepth.shape

    # Initialize dose matrix
    dose_in_pat_matrix = np.zeros(ct_shape)

    # Get voxel size in mm (2.5mm)
    voxel_size = tp_plan_obj.voxelsize  # float, in mm (e.g., 2.5 mm)

    # Adjust to convention 0°= up, 90°=right, 180°=down, 270°=left (clockwise)
    # from convention: 0°=right, 90°=up, 180°=left, 270°=down (counter-clockwise)
    angle_math = 270 - angle
    angle_rad = np.radians(angle_math)

    # Get isocenter position (reference point for beam)
    isocenter = get_isocenter(tp_plan_obj=tp_plan_obj, isocenter_method='com_tumor')

    # =====================================================================
    # Calculate dose for each voxel in the CT slice
    # =====================================================================
    # Loop over each voxel
    for i in range(ct_shape[0]):  # Loop over rows (y-direction)
        for j in range(ct_shape[1]):  # Loop over columns (x-direction)

            # Get radiological depth for this voxel
            zrad = raddepth[i, j]

            # Look up beam parameters at this raddepth
            # Find the closest tabulated depth value in our proton dose Monte Carlo data
            # Could also perfom interpolation
            depth_idx = np.argmin(np.abs(depth_mm - zrad))

            # Retrieve D0 and sigma at this depth from lookup table
            D0_val = D0[depth_idx]  # Central axis dose at depth zrad
            sigma_val = sigma[depth_idx]  # Beam width (std dev) at depth zrad

            # Skip if sigma is very low, (sigma ≈ 0 indicates invalid data point)
            if sigma_val <= 1e-10:
                continue

            # ---------------------------------------------------------
            # Convert voxel position to physical coordinates
            # ---------------------------------------------------------
            # Convert from voxel indices (i,j) to physical position in mm
            voxel_y = i * voxel_size  # Row index → y-coordinate (mm)
            voxel_x = j * voxel_size  # Column index → x-coordinate (mm)

            # Convert isocenter from voxel indices to physical coordinates
            iso_y = isocenter.get('y') * voxel_size  # mm
            iso_x = isocenter.get('x') * voxel_size  # mm

            # Calculate position relative to isocenter (origin of beam coordinate system)
            rel_x = voxel_x - iso_x  # mm
            rel_y = voxel_y - iso_y  # mm

            # ---------------------------------------------------------
            # Transform to beam coordinate system
            # ---------------------------------------------------------
            # The beam coordinate system has:
            # - y-axis along the beam direction (direction of propagation)
            # - x-axis perpendicular to beam (lateral direction)
            # - z-axis out of plane (set to 0 for 2D and dont take into account)

            # We need the perpendicular distance from the beam's central axis
            # Rotation formula: x_beam = -x*sin(θ) + y*cos(θ)
            x_beam = -rel_x * np.sin(angle_rad) + rel_y * np.cos(angle_rad)

            # Apply lateral beam offset (if beam center is not at isocenter)
            # latpos['x'] shifts the beam laterally in the beam coordinate system
            x_beam_shifted = x_beam - latpos['x']

            # ---------------------------------------------------------
            # Calculate dose using Gaussian pencil beam model
            # ---------------------------------------------------------
            # Lateral distance squared from beam axis (for y=0 in 2D)
            lateral_distance_squared = x_beam_shifted ** 2

            # Gaussian dose distribution formula:
            # D(x) = D0 * [1/(2πσ²)] * exp[-x²/(2σ²)]
            #
            # Components:
            # 1. D0_val: central axis dose at this depth
            # 2. normalization: ensures proper integral over lateral dimension
            # 3. exponential: Gaussian falloff with lateral distance

            normalization = 1.0 / (2.0 * np.pi * sigma_val ** 2)
            exponential = np.exp(-lateral_distance_squared / (2.0 * sigma_val ** 2))
            dose = D0_val * normalization * exponential

            # Store calculated dose in the patient dose matrix
            dose_in_pat_matrix[i, j] = dose

    return {
        "angle": angle,  # Beam angle
        "energy": init_energy,  # Initial proton energy (MeV)
        "latpos": latpos,  # Lateral position offset (mm)
        "dose": dose_in_pat_matrix  # 2D dose distribution
    }


def show_proton_pencil_beam_dose_on_ct(
        tp_plan_path: Path = str(project_root_provider()) + r'.\utils\data\patientdata.mat',
        protondosesfolder_path: Path = str(project_root_provider()) + r'.\utils\data\protondosedata',
        initial_energy: float = 0.0,  # MeV
        angle: float = 0.0,
        latpos: Dict[str, float] = None,
        voinames_colors_visualization: List[Tuple[str]] = None):
    """
    Visualize proton pencil beam dose distribution on CT.

    Parameters:
    -----------
    tp_plan_path : Path
        Path to treatment plan data file
    protondosesfolder_path : Path
        Path to folder containing proton dose data files
    initial_energy : float
        Initial proton energy in MeV
    angle : float
        Beam angle in degrees
    latpos : dict
        Lateral position of beam {'x': float, 'y': float}
    voinames_colors_visualization : list
        List of VOI names and colors for visualization
    """

    if latpos is None:
        latpos = {'x': 0.0, 'y': 0.0}

    show_outside_body = False

    # Get CT data
    tp_plan_obj, _ = visualize_tp_plan_data(
        tp_plan_path=tp_plan_path,
        show_plot=False,
        voinames_colors_visualization=voinames_colors_visualization
    )

    # Get proton beam data
    # Construct path to proton beam dose file based on init energy
    protondosesfile_path = Path(protondosesfolder_path) / Path(f'pbmcs{str(float(initial_energy))}.dat')

    # Parse proton dose data
    protondose_data = get_proton_dose_data(protondosesfile_path=protondosesfile_path)

    # Get isocenter
    isocenter = get_isocenter(tp_plan_obj=tp_plan_obj, isocenter_method='com_tumor')

    # Get Raddepth matrix from ct for angle, voxel size and step size
    raddepth_ct = calculate_raddepth(
        ct_scan=tp_plan_obj.ct,
        isocenter=isocenter,
        pixel_size=tp_plan_obj.voxelsize,
        step_size=0.2,
        angle=angle
    )

    # Compute dose distribution using pencil beam algorithm
    pb_result = calculate_proton_pencil_beam_dose(
        angle=angle,
        init_energy=initial_energy,
        latpos=latpos,
        raddepth=raddepth_ct,
        tp_plan_obj=tp_plan_obj,
        protondose_data=protondose_data
    )

    # Remove values outside pat
    if not show_outside_body:
        body_mask = get_body_mask(tp_plan_obj=tp_plan_obj)
        outside_body = (body_mask == 0)
        beam_dose = np.where(outside_body, 0, pb_result.get('dose'))

    # Remove values below chosen percentage
    remove_below_percentage = 1

    # Remove/mask dose values below cutoff
    beam_dose = np.ma.masked_where(beam_dose < ((remove_below_percentage / 100) * beam_dose.max()), beam_dose)

    # Visualize dose distribution on CT
    ct_img = plt.imshow(tp_plan_obj.ct,origin='lower', cmap='grey')
    protondose_img = plt.imshow(beam_dose,origin='lower', cmap='jet', alpha=0.5)
    plt.legend()
    plt.title(f"E = {initial_energy} MeV, angle = {angle} deg, latpos = {latpos['x']}")
    plt.colorbar(protondose_img, label=f'proton beam dose (Gy)')

    plt.show()

# #=====================================Week7===============================

# New calculate pencil beam dose implementation for photons


def get_equispaced_full_rot_angles(number_of_angles:int=9)->List[float]:
    """Get list of angles by deviding 360 by number of angles and then multiplying
    from 0 to number of angles and saving each result"""
    division_angle = 360/number_of_angles
    return [division_angle*x for x in range(number_of_angles)]

def create_photon_beams(angles: List[float], tp_plan_path:str = str(project_root_provider()) + r'.\utils\data\patientdata.mat',
                        beamletdose_path: str = str(project_root_provider()) + r'.\utils\data\photondosedata\beamletdose5mm.mat',
                        show_beams = False) -> Dict[str, any]:
    # prepare beams container
    beams = {}
    beamlet_positions = [-12,-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10,12]

    # Get CT data.
    tp_plan_obj, _ = visualize_tp_plan_data(tp_plan_path=tp_plan_path,
                                            show_plot=False,
                                            voinames_colors_visualization=None)

    # Get beamlet data
    beamletdose_obj, beamlet_img = visualize_beamletdose_data(beamletdose_path=beamletdose_path,
                           show_plot=False)

    # Interpolate beamlet to ct grid
    beamlet_dose_on_ct_grid = interpolate_to_grid(from_img_data=beamletdose_obj.dose,
                                                  from_voxelsize=beamletdose_obj.voxelsize,
                                                  to_voxelsize=tp_plan_obj.voxelsize)

    # Get isocenter (Center of Mass tumor)
    isocenter = get_isocenter(tp_plan_obj, 'COM_tumor')

    beamNo = 0

    for angle in angles:
        beamNo +=1
        # Get Raddepth matrix from ct for angle, boxelsize and stepsize
        raddepth = calculate_raddepth(ct_scan=tp_plan_obj.ct, isocenter=isocenter,
                                         pixel_size=tp_plan_obj.voxelsize, step_size=0.2, angle=angle)
        pb = {}

        # Get pencil beam dose
        for beamletpos in beamlet_positions:
            pb_instance = calculate_pencil_beam_dose(beamlet_dose_on_ct_grid, angle,
                                            beamletpos, isocenter, raddepth,
                                            voxelsize=tp_plan_obj.voxelsize)
            pb_key = f'beamlet_latpos{beamletpos}'
            pb[pb_key] = pb_instance

        beams[f'{beamNo}'] = {'angle': angle, 'nBeamlets': len(beamlet_positions), 'beamletpos': beamlet_positions, 'raddepth': raddepth, 'pb': pb}

    if show_beams:
        # Plot CT once
        plt.imshow(tp_plan_obj.ct, origin='lower', cmap='gray')

        # Combine all dose arrays before plotting
        dose_sum = np.zeros_like(tp_plan_obj.ct, dtype=float)
        for beam in beams:
            pb = beams[beam]['pb']
            beamlet_positions = beams[beam]['beamletpos']
            for beamletpos in beamlet_positions:
                pb_key = f'beamlet_latpos{beamletpos}'
                pb_instance = pb[pb_key]
                dose_sum += pb_instance.get('dose')

        # Plot combined dose once
        plt.imshow(dose_sum, origin='lower', cmap='jet', alpha=0.5)
        plt.colorbar(label='Total dose (Gy)')

        plt.show()
    return beams

def create_dij_matrix(beams: Dict[str, any]) -> np.ndarray:
    """Create D_ij matrix from beams dictionary and treatment plan object"""
    # Get size of the dose arrays
    nVoxels = beams['1']['pb'][list(beams['1']['pb'])[0]]['dose'].size
    nBeams = len(beams)
    nBeamletsPerBeam = beams['1']['nBeamlets']  # Assuming all beams have same number of beamlets
    nTotalBeamlets = nBeams * nBeamletsPerBeam

    # Initialize D_ij matrix
    Dij = np.zeros((nVoxels, nTotalBeamlets))

    beamletIndex = 0
    for beam in beams:
        pb = beams[beam]['pb']
        beamlet_positions = beams[beam]['beamletpos']
        for beamletpos in beamlet_positions:
            pb_key = f'beamlet_latpos{beamletpos}'
            pb_instance = pb[pb_key]
            # dose_array = pb_instance.get('dose').flatten()  # Flatten to 1D array
            dose_array = numpy.reshape(pb_instance.get('dose'), nVoxels)  # Flatten to 1D array

            # Fill corresponding column in D_ij matrix
            Dij[:, beamletIndex] = dose_array
            beamletIndex += 1

    # Create uniform fluence vector (1 for each beamlet)
    uniform_fluence_vector = np.ones((nTotalBeamlets, 1))
    return Dij, uniform_fluence_vector

def get_dose_from_dijs_and_fluence_vector(dij_matrix: np.ndarray, fluence_vector: np.ndarray) -> np.ndarray:
    """Calculate dose distribution from D_ij matrix and fluence vector"""
    # Matrix multiplication to get dose distribution
    dose_array = np.dot(dij_matrix, fluence_vector)

    return dose_array

def plot_dij_matrix_on_ct(dij_matrix_dose, show_plot:bool=True,
                          tp_plan_path:str =str(project_root_provider()) + r'.\utils\data\patientdata.mat',
                          plot_note:str = '', voinames_colors_visualization=[('tumor','green')]):
    """Plot a specific column of D_ij matrix on CT"""
    # Get CT data.
    tp_plan_obj, _ = visualize_tp_plan_data(tp_plan_path=tp_plan_path,
                                            show_plot=False,
                                            voinames_colors_visualization=voinames_colors_visualization)

    dose_sum = dij_matrix_dose.reshape(tp_plan_obj.ct.shape)

    # Visualize dose distribution on CT
    ct_img = plt.imshow(tp_plan_obj.ct, origin='lower', cmap='grey')
    dose_img = plt.imshow(dose_sum, origin='lower', cmap='jet', alpha=0.5)
    plt.title(f'Dose on CT: {plot_note}')
    plt.colorbar(dose_img, label='Dose (Gy)')

    if show_plot:
        plt.show()

# #=====================================Week8===============================


def get_vois_dij_format(tplan_obj, dij_matrix) -> Dict[str,np.array]:
    voinames_in_dij_format = {}
    voinumbers = get_voinames_number(tplan_obj.voinames)

    # Flatten VOI matrix to match dij_matrix shape
    voi_in_dij_format = tplan_obj.voi.reshape(-1)

    for voi_name, voi_number in voinumbers.items():
        # Create boolean mask for this VOI
        if not voinames_in_dij_format.get(f'{voi_name}'): voinames_in_dij_format[voi_name] = {}
        voinames_in_dij_format[voi_name]['1d_mask'] = (voi_in_dij_format == voi_number)

    return voinames_in_dij_format




def define_plan_objectives(tplan_obj: TPlan, voinumbers: Dict[str,int])-> Dict[str, List[Dict]]:
    """
    returns a dict of 'vois' which is a list of individual dicts of the following format:
    {
            "name": "normal tissue",
            "nVoxels": np.sum(tplan_obj.voi == voinumbers.get('normal tissue')),
            "maxdose": 20,
            "overdosepenalty": 7,
            "mindose": 0,
            "underdosepenalty": 0,
            "minconstraint": 10,
            "maxconstraint": 25,
    }

    """

    TPopt = {"vois": [
        {
            "name": "normal tissue",
            "nVoxels": np.sum(tplan_obj.voi == voinumbers.get('normal tissue')),
            "maxdose": 20,
            "overdosepenalty": 7,
            "mindose": 0,
            "underdosepenalty": 0,
        },
        {
            "name": "right lung",
            "nVoxels": np.sum(tplan_obj.voi == voinumbers.get('right lung')),
            "maxdose": 15,
            "overdosepenalty": 10,
            "mindose": 0,
            "underdosepenalty": 0,
            # "maxconstraint": 10,
        },
        {
            "name": "left lung",
            "nVoxels": np.sum(tplan_obj.voi == voinumbers.get('left lung')),
            "maxdose": 15,
            "overdosepenalty": 10,
            "mindose": 0,
            "underdosepenalty": 0,
        },
        {
            "name": "spinal cord",
            "nVoxels": np.sum(tplan_obj.voi == voinumbers.get('spinal cord')),
            "maxdose": 15,
            "overdosepenalty": 40,
            "mindose": 0,
            "underdosepenalty": 0,
            "maxconstraint":20,
        },
        {
            "name": "esophagus",
            "nVoxels": np.sum(tplan_obj.voi == voinumbers.get('esophagus')),
            "maxdose": 15,
            "overdosepenalty": 10,
            "mindose": 0,
            "underdosepenalty": 0,
        },
        {
            "name": "tumor",
            "nVoxels": np.sum(tplan_obj.voi == voinumbers.get('tumor')),
            "maxdose": 50,
            "overdosepenalty": 1,
            "mindose": 35,
            "underdosepenalty": 40,
            "minconstraint": 15,
            "maxconstraint": 55,
        }
    ]
    }

    return TPopt


def get_voi_variables_and_dose_for_objectives(dose, TPopt, voiname):
    mask = None
    for voi in TPopt['vois']:
        if voi['name'] == voiname:
            mask, maxdose, overdosepenalty, mindose, underdosepenalty, nvoxels = voi['1d_mask'], voi['maxdose'], voi['overdosepenalty'], voi['mindose'], voi['underdosepenalty'], voi['nVoxels']
            break
    if mask.any() == None:
        print('An error has occurred getting the voi mask for the objective function')
    voi_dose = dose[mask]
    return voi_dose, maxdose, overdosepenalty, mindose, underdosepenalty, nvoxels, mask

def calculate_quadratic_objective(bixelweights: np.ndarray, TPopt: Dict, dij_matrix:np.ndarray, voiname:str = ''):
    """
    Calculate the quadratic objective function for IMRT optimization.

    The objective function is:
    f(d) = Σ w^o_i * (d_i - D^max_i)²_+ + Σ w^u_i * (D^min_i - d_i)²_+

    The first term penalizes doses that exceed max dose, the second term penalizes doses below min dose

    Params:
    -----------
    bixelweights : Vector of beamlet intensities (bixel weights)
    TPopt : Dictionary containing optimization parameters

    Return:
    --------
    objective_value : The value of the quadratic objective function
    """

    # Calculate dose vector: d = A * w
    # where A is the dose matrix and w is the bixel weights
    dose = dij_matrix @ bixelweights

    # use boolean mask in TPopt, to filter only the relevant voxels for the objective value
    voi_dose, maxdose, overdosepenalty, mindose, underdosepenalty,nvoxels,_ = get_voi_variables_and_dose_for_objectives(dose, TPopt, voiname)

    # Normalize by num voxels in voi
    overdosepenalty, underdosepenalty = overdosepenalty/nvoxels, underdosepenalty/nvoxels

    # Calculate overdose term: w^o_i * (d_i - D^max_i)²_+
    # Only penalize when dose exceeds maximum (positive part)
    overdose_diff = voi_dose - maxdose
    overdose_term = overdosepenalty * np.maximum(0, overdose_diff) ** 2

    # Calculate underdose term: w^u_i * (D^min_i - d_i)²_+
    # Only penalize when dose is below minimum (positive part)
    underdose_diff = mindose - voi_dose
    underdose_term = underdosepenalty * np.maximum(0, underdose_diff) ** 2

    # Sum all penalty terms
    voi_objective_value = np.sum(overdose_term) + np.sum(underdose_term)

    return voi_objective_value

def calculate_quadratic_objective_gradient(bixelweights: np.ndarray, TPopt: Dict,
                                           dij_matrix: np.ndarray, voiname: str = ''):
    """Calculate gradient of f for gradient descent for IMRT optimization."""

    # Calculate dose vector: d = A * w
    # where A is the dose matrix and w is the bixel weights
    dose = dij_matrix @ bixelweights

    # use boolean mask in TPopt, to filter only the relevant voxels for the objective value
    voi_dose, maxdose, overdosepenalty, mindose, underdosepenalty, nvoxels, mask = get_voi_variables_and_dose_for_objectives(dose, TPopt, voiname)


    # Normalize by num voxels in voi
    overdosepenalty, underdosepenalty = overdosepenalty/nvoxels, underdosepenalty/nvoxels

    # Calculate gradient w.r.t. dose (∂f/∂d)
    # Initialize with zeros (gradient is zero where constraints aren't violated)
    dose_gradient = np.zeros_like(dose)

    # Overdose component: 2·wᵒ·(d - Dᵐᵃˣ) when d > Dᵐᵃˣ
    # The mask (overdose_diff > 0) implements the positive part function (·)₊
    overdose_diff = voi_dose - maxdose
    overdose_mask = overdose_diff > 0  # Only where dose exceeds max
    overdose_gradient = overdosepenalty * overdose_diff * overdose_mask

    # Underdose component: -2·wᵘ·(Dᵐⁱⁿ - d) when d < Dᵐⁱⁿ
    # Note the negative sign! Gradient points toward increasing dose
    underdose_diff = mindose - voi_dose
    underdose_mask = underdose_diff > 0  # Only where dose below min
    underdose_gradient = -underdosepenalty * underdose_diff * underdose_mask

    # Combine contributions (only for voxels in this VOI)
    dose_gradient[mask] = overdose_gradient + underdose_gradient

    # Backpropagate through dose matrix using chain rule
    # ∇f = (∂f/∂d)·(∂d/∂w) = dose_gradient · Aᵀ
    # The factor of 2 comes from the derivative of x²
    gradient = 2 * dij_matrix.T @ dose_gradient

    return gradient


def optimize_fluence_for_objectives(dij_matrix: np.array, fluence_matrix: np.array,
                                    objectives: Dict, method:str = 'simple_gradient_descent', iterations:int=1000,
                                    step_size:float=0.0005):

    print(f'Optimizing dose with {method}. Chosen iterations: {iterations} with stepsize: {step_size}')


    # initialize fluence matrix to zeros
    fluence_matrix[:] = 0
    all_iterations = {}
    print_info = (iterations)/3


    for i in range(iterations):
        if i % round(print_info)==0 and i != 0:
            prev_i = i-1
            print("=" * 70)
            print(f'At iteration {i}/{iterations} in objective optimization.')
            print(f'Total objective value: {all_iterations[f"{prev_i}"]["combined_obj_val"]}')
            print(f'Total gradient values: {all_iterations[f"{prev_i}"]["combined_gradient"]}')
            print(f'Last individual objective values: {all_iterations[f"{prev_i}"]["obj_vals"]}')
        # Initialize values as empty or zeros
        obj_values = []
        total_objective = 0.0
        total_gradient = np.zeros(len(fluence_matrix))

        # For each voi
        for voi in objectives['vois']:
            # Get quadratic objective value
            voi_quadr_objective_value = calculate_quadratic_objective(fluence_matrix, objectives, dij_matrix, voi['name'])
            # Get beam gradient values
            voi_quadr_objective_gradient = calculate_quadratic_objective_gradient(fluence_matrix, objectives, dij_matrix, voi['name'])
            # Collapse into a single objective metric for all vois
            total_objective += voi_quadr_objective_value
            # Collapse into a single gradient metric for all vois
            total_gradient += voi_quadr_objective_gradient.flatten()
            # Track data
            voi_objective_params = {'name' :voi['name'], 'obj_value':voi_quadr_objective_value, 'obj_gradient': voi_quadr_objective_gradient}
            obj_values.append(voi_objective_params)
        all_iterations[f'{i}'] = {'obj_vals':obj_values,'combined_obj_val':total_objective,'combined_gradient':total_gradient}

        # Update fluence using gradient descent
        # w_new = w_old - α * gradient of f
        fluence_matrix = fluence_matrix.flatten() - step_size * total_gradient.flatten()

        # Fluence weights cannot be negative (we cannot suck up dose :))
        fluence_matrix = np.maximum(0, fluence_matrix)

    return fluence_matrix, all_iterations

def plot_opti_history(opti_logs):
    t = 0
    t_list=[]
    y_list=[]
    for log in opti_logs:
        y = opti_logs[f'{log}']['combined_obj_val']
        t+=1
        y_list.append(y)
        t_list.append(t)


    fig,(plt1,plt2) = plt.subplots(1,2)
    fig.suptitle("objective values vs iterations")
    fig.supylabel('objective value')
    fig.supxlabel('iterations')
    plt1.plot(t_list,y_list)
    plt2.semilogy(t_list,y_list)
    plt.show()


    # #=====================================Week9===============================


def optimize_photons_with_scipy(dij_matrix: np.array, fluence_matrix: np.array,
                                objectives: Dict, max_iterations: int = 1000, ftol=1e-6):
    print(f'Optimizing dose with scipy SLSQP. Max iterations: {max_iterations}')

    # Initialize fluence matrix to zeros
    initial_fluence = np.zeros_like(fluence_matrix.flatten())

    # Track iteration history
    iteration_history = []

    # Define combined objective function (sums over all VOIs)
    def combined_objective(fluence_weights):
        """Single objective function that sums objectives from all VOIs"""
        total_objective = 0.0
        for voi in objectives['vois']:
            voi_obj = calculate_quadratic_objective(
                fluence_weights, objectives, dij_matrix, voi['name']
            )
            total_objective += voi_obj
        return total_objective

    # Define combined gradient function (sums gradients from all VOIs)
    def combined_gradient(fluence_weights):
        """Single gradient function that sums gradients from all VOIs"""
        total_gradient = np.zeros_like(fluence_weights)
        for voi in objectives['vois']:
            voi_grad = calculate_quadratic_objective_gradient(
                fluence_weights, objectives, dij_matrix, voi['name']
            )
            total_gradient += voi_grad.flatten()
        return total_gradient


# These constrain the DOSE
    def create_dose_constraints():
        """
        Create hard dose constraints for each VOI.
        NOTE: These are HARD constraints - dose MUST satisfy them.
        Not soft constraints achieved through objectives

        To use hard constraints, add these fields to your VOI dicts:
        - 'maxconstraint': absolute maximum dose allowed (Gy)
        - 'minconstraint': absolute minimum dose allowed (Gy)
        """
        constraints = []

        for voi in objectives['vois']:
            voi_name = voi['name']

            # Check if this VOI has hard constraint requirements
            hard_max = voi.get('maxconstraint', None)
            hard_min = voi.get('minconstraint', None)

            # Skip if no hard constraints specified
            if hard_max is None and hard_min is None:
                continue

            # Maximum dose constraint: max_dose - dose[voi_voxels] >= 0
            if hard_max is not None:
                def max_dose_constraint_func(fluence_weights, name=voi_name, max_val=hard_max):
                    dose = dij_matrix @ fluence_weights
                    voi_dose, _, _, _, _, _, _ = get_voi_variables_and_dose_for_objectives(
                        dose, objectives, name
                    )
                    # Return array: max_val - voi_dose (positive when constraint satisfied)
                    return max_val - voi_dose

                constraints.append({
                    'type': 'ineq',
                    'fun': max_dose_constraint_func
                })
                print(f"  Added hard max dose constraint for {voi_name}: {hard_max} Gy")

            # Minimum dose constraint: dose[voi_voxels] - min_dose >= 0
            if hard_min is not None:
                def min_dose_constraint_func(fluence_weights, name=voi_name, min_val=hard_min):
                    dose = dij_matrix @ fluence_weights
                    voi_dose, _, _, _, _, _, _ = get_voi_variables_and_dose_for_objectives(
                        dose, objectives, name
                    )
                    # Return array: voi_dose - min_val (positive when constraint satisfied)
                    return voi_dose - min_val

                constraints.append({
                    'type': 'ineq',
                    'fun': min_dose_constraint_func
                })
                print(f"  Added hard min dose constraint for {voi_name}: {hard_min} Gy")

        return constraints if constraints else None
    # Bounds: fluence must be non-negative
    bounds = [(0, None) for _ in range(len(initial_fluence))]

    # Create constraints
    constraints = create_dose_constraints()

    # Run optimization (single call!)
    print("Starting optimization...")
    result = scipy.optimize.minimize(
        fun=combined_objective,  # Function to minimize
        x0=initial_fluence,  # Initial guess
        method='SLSQP',  # Method
        jac=combined_gradient,  # Gradient function
        bounds=bounds,  # Non-negative constraint
        constraints=constraints,  # Dose constraints (if any)
        options={
            'maxiter': max_iterations,
            'disp': True,  # Display convergence messages
            'ftol': ftol  # Function tolerance value
        }
    )

    print("=" * 70)
    print("Optimization complete!")
    print(f"Final objective value: {result.fun:.6f}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Iterations: {result.nit}")

    # Get optimized fluence
    optimized_fluence = result.x.reshape(fluence_matrix.shape)

    # Compile final iteration data
    final_iteration_data = {
        'optimized_fluence': optimized_fluence,
        'total_objective': result.fun,
        'gradient_norm': np.linalg.norm(result.jac),
        'success': result.success,
        'message': result.message,
        'iterations': result.nit,
    }
    return optimized_fluence, final_iteration_data, iteration_history

def create_proton_beams(angles: List [float],tp_plan_path: Path = str(project_root_provider()) + r'.\utils\data\patientdata.mat',
                        protondose_path:Path = str(project_root_provider()) + r'.\utils\data\protondosedata', show_beams = False):
    # prepare beams container
    beams = {}
    beamlet_positions = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12]

    # Get CT data.
    tp_plan_obj, _ = visualize_tp_plan_data(tp_plan_path=tp_plan_path,
                                            show_plot=False,
                                            voinames_colors_visualization=None)

    # Get proton beam data
    # Construct path to proton beam dose file based on init energy
    # protondosesfile_path = Path(protondosesfolder_path) / Path(f'pbmcs{str(float(initial_energy))}.dat')
    protondosesavailenergies_path = Path(protondose_path) / Path(f'edata.dat')
    protondosesenergiesinfo_path = Path(protondose_path) / Path(f'rdata.dat')

    proton_energies_list = numpy.loadtxt(protondosesavailenergies_path)
    proton_energies_info_list = numpy.loadtxt(protondosesenergiesinfo_path)

    # Parse proton dose data
    # protondose_data = get_proton_dose_data(protondosesfile_path=protondosesfile_path)

    # Get isocenter
    isocenter = get_isocenter(tp_plan_obj=tp_plan_obj, isocenter_method='com_tumor')

    beamNo = 0
    for angle in angles:
        beamNo +=1
        # Get Raddepth matrix from ct for angle, voxel size and step size
        raddepth_ct = calculate_raddepth(
            ct_scan=tp_plan_obj.ct,
            isocenter=isocenter,
            pixel_size=tp_plan_obj.voxelsize,
            step_size=0.2,
            angle=angle
        )

        # for each beamletpos, check if tumor is in path.
        # if tumor is in path get max and min raddepth within tumor mask on the latpos path
        # energies_per_latpos = get_energies_beamletpos()

        # for latpos in energies_per_latpos:
        #     for energy in latpos:
        #         # Compute dose distribution using pencil beam algorithm
        #         pb_result = calculate_proton_pencil_beam_dose(
        #             angle=angle,
        #             init_energy=initial_energy,
        #             latpos=latpos,
        #             raddepth=raddepth_ct,
        #             tp_plan_obj=tp_plan_obj,
        #             protondose_data=protondose_data
        #         )
        # beams[f'{beamNo}'] = {'angle': angle, 'nBeamlets': len(beamlet_positions), 'beamletpos': beamlet_positions, 'raddepth': raddepth, 'pb': pb}

    return beams

if __name__ == '__main__':
    # #=====================================Week1===============================
    # Visualize CT with contours

    # # CT with contours visualization
    # visualize_tp_plan_data(
    #     tp_plan_path=Path(str(project_root_provider()) + r'.\utils\data\patientdata.mat'),
    #     voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','blue')],
    #     show_plot=True)

    # #=====================================Week2===============================
    # Visualize dose on CT with contours

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
    # Radiological depth calculation and visualization on ct

    # raddepth_on_ct(
    # tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     angle=60,
    #     isocenter_method='com_body'
    # )

    # raddepth_on_ct(
    # tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     angle=100,
    #     isocenter_method='com_tumor',
    #     step_size=0.2,
    #     show_outside_body=False
    # )
    #
    # raddepth_on_ct(
    # tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     angle=200,
    #     isocenter_method='com_tumor',
    #     step_size=0.2,
    #     show_outside_body=False
    # )
    #
    # raddepth_on_ct(
    # tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     angle=300,
    #     isocenter_method='com_tumor',
    #     step_size=0.2,
    #     show_outside_body=False
    # )

    # #=====================================Week4===============================
    # Photon pencil beam dose in patient

    # show_pencil_beam_dose_on_ct(
    #     tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     beamletdose_path=str(project_root_provider()) + r'.\utils\data\photondosedata\beamletdose5mm.mat',
    #     angle=45,
    #     latpos={'x':30,'y':0},
    #     voinames_colors_visualization=[('tumor', 'purple'),('esophagus','magenta'),('spinal cord','brown')],
    #
    # )
    #
     # show_pencil_beam_dose_on_ct(
     #     tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
     #     beamletdose_path=str(project_root_provider()) + r'.\utils\data\photondosedata\beamletdose5mm.mat',
     #     angle=90,
     #     latpos={'x':10,'y':0},
     #     voinames_colors_visualization=[('tumor', 'purple'),('esophagus','magenta'),('spinal cord','brown')],
     # )

    # #=====================================Week5===============================
    # Bethe Bloch depth-dose calculation

    #Generate data and solve for 100MeV
    # calculate_and_plot_residual_e_and_de_dz_with_and_without_range_straggling(energies_to_test=[50,100,300])

    # Work on summary
    # 0/ introduction
    # 1. energy fluence
    # 2. Classical bethe bloch
    # 3. Relativistic bethe bloch
    # 4. Stopping power
    # - residual energy as a function of depth
    # - stopping power in terms of depth
    # 5. Range straggling

    # #=====================================Week6===============================
    # Proton pencil beam dose in patient
    # Available proton dose data files can be found in the utils\data\protondosedata folder

    # # Example of Noemi
    # show_proton_pencil_beam_dose_on_ct(
    #     tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     protondosesfolder_path=str(project_root_provider()) + r'.\utils\data\protondosedata',
    #     initial_energy=87.7,  # MeV
    #     angle=60,
    #     latpos={'x':40,'y':0},
    #     voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','orange')],
    # )
    #
    # # Case in the heart, could represent water
    # show_proton_pencil_beam_dose_on_ct(
    #     tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     protondosesfolder_path=str(project_root_provider()) + r'.\utils\data\protondosedata',
    #     initial_energy=87.7,  # MeV
    #     angle=180,
    #     latpos={'x':-60,'y':0},
    #     voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','orange')],
    # )
    #
    # # Case through the lung
    # show_proton_pencil_beam_dose_on_ct(
    #     tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     protondosesfolder_path=str(project_root_provider()) + r'.\utils\data\protondosedata',
    #     initial_energy=87.7,  # MeV
    #     angle=180,
    #     latpos={'x':20,'y':0},
    #     voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','orange')],
    # )
    #
    # # Beam through rib air tissue interface
    # show_proton_pencil_beam_dose_on_ct(
    #     tp_plan_path=str(project_root_provider()) + r'.\utils\data\patientdata.mat',
    #     protondosesfolder_path=str(project_root_provider()) + r'.\utils\data\protondosedata',
    #     initial_energy=87.7,  # MeV
    #     angle=90,
    #     latpos={'x':-9,'y':0},
    #     voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','orange')],
    # )

    # #=====================================Week7===============================
    # Use different beam angles and dijs matrices to create dose distribution

    # # # Get a list of equispaced beam angles, use an uneven number to avoid symmetry
    # angles = get_equispaced_full_rot_angles(9)
    # # Use the angles to get the beams with beamlets and beamlet positions (lateral offsets)
    # beams = create_beams(angles)
    # # Create dose influence matrix for all beams and beamlets
    # dij_matrix, uniform_fluence_vector = create_dij_matrix(beams)
    #
    # # Calculate dose distribution for uniform fluence
    # scaled_dij_matrix = get_dose_from_dijs_and_fluence_vector(dij_matrix, uniform_fluence_vector)
    # # Plot dose distribution for uniform fluence
    # plot_dij_matrix_on_ct(scaled_dij_matrix, plot_note='Uniform fluence')
    #
    # # Plot dose distribution for specific beamlet indices to find beamlet of each angle hitting spinal cord
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=7)
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=20)
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=33)
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=46)
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=57)
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=58)
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=70)
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=83)
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=96)
    # # plot_dij_matrix_on_ct(dij_matrix,beamlet_index=110)
    #
    # # Create modified fluence vector to spare spinal cord
    # uniform_fluence_vector_edit = uniform_fluence_vector.copy()
    # # Zero out beamlets that hit spinal cord (based on previous plots)
    # beamlets_fluence_to_zero = [7,20,33,46,57,58,70,83,96,110]
    # for beamlet in beamlets_fluence_to_zero:
    #     uniform_fluence_vector_edit[beamlet] = 0.0
    # # Calculate dose distribution for modified fluence
    # scaled_dij_matrix_simple_edit = get_dose_from_dijs_and_fluence_vector(dij_matrix, uniform_fluence_vector_edit)
    # # Plot dose distribution for modified fluence
    # plot_dij_matrix_on_ct(scaled_dij_matrix_simple_edit, plot_note='Spare spine')

    # #=====================================Week8===============================
    # Implement gradient descent and use it to optimize for convex objectives to do IMRT

    # Still need to fix photon interpolation implementation creating
    # low resolution dose distribution with 'holes' after rotation

#     tplan_obj, _ = visualize_tp_plan_data(
#         tp_plan_path=Path(str(project_root_provider()) + r'.\utils\data\patientdata.mat'),
#         voinames_colors_visualization=None,
#         show_plot=False)
#
#     voinumbers = get_voinames_number(tplan_obj.voinames)
#
#     TPopt_objectives = define_plan_objectives(tplan_obj, voinumbers)
#
#
# # # Get a list of equispaced beam angles, use an uneven number to avoid symmetry
#     nbeams = 7
#     angles = get_equispaced_full_rot_angles(nbeams)
#     # Use the angles to get the beams with beamlets and beamlet positions (lateral offsets)
#     beams = create_beams(angles)
#     # Create dose influence matrix for all beams and beamlets
#     dij_matrix, uniform_fluence_vector = create_dij_matrix(beams)
#
#     # Calculate dose distribution for uniform fluence
#     scaled_dij_matrix = get_dose_from_dijs_and_fluence_vector(dij_matrix, uniform_fluence_vector)
#
#     # Get the voi masks in the di matrix format
#     vois_in_dij_format = get_vois_dij_format(tplan_obj,scaled_dij_matrix)
#
#     # Add vois in dij format to tpopt_objectives
#     for count, voi in enumerate(TPopt_objectives['vois']):
#         dij_format_vois = vois_in_dij_format[f"{voi['name']}"]['1d_mask']
#         TPopt_objectives['vois'][count]['1d_mask'] = dij_format_vois
#
#     # Optimize for plan objective
#     opti_fluence, opti_logs = optimize_fluence_for_objectives(dij_matrix=dij_matrix,fluence_matrix=uniform_fluence_vector,objectives=TPopt_objectives )
#
#     # Get dose for fluences
#     dose = get_dose_from_dijs_and_fluence_vector(dij_matrix, opti_fluence)
#
#     # Plot optimization hisotry
#     plot_opti_history(opti_logs)
#
#     # Plot dose on ct
#     plot_dij_matrix_on_ct(dose, plot_note=f'Optimized fluence, Nbeam angles: {nbeams}', voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','orange')])

# #=====================================Week9===============================
# Perform optimization with scipy solver for IMRT

    #     tplan_obj, _ = visualize_tp_plan_data(
    #         tp_plan_path=Path(str(project_root_provider()) + r'.\utils\data\patientdata.mat'),
    #         voinames_colors_visualization=None,
    #         show_plot=False)
    #
    #     voinumbers = get_voinames_number(tplan_obj.voinames)
    #
    #     TPopt_objectives = define_plan_objectives(tplan_obj, voinumbers)
    #
    # # # Get a list of equispaced beam angles, use an uneven number to avoid symmetry
    #     nbeams = 9
    #     angles = get_equispaced_full_rot_angles(nbeams)
    #     # Use the angles to get the beams with beamlets and beamlet positions (lateral offsets)
    #     beams = create_beams(angles)
    #     # Create dose influence matrix for all beams and beamlets
    #     dij_matrix, uniform_fluence_vector = create_dij_matrix(beams)
    #
    #     # Calculate dose distribution for uniform fluence
    #     scaled_dij_matrix = get_dose_from_dijs_and_fluence_vector(dij_matrix, uniform_fluence_vector)
    #
    #     # Get the voi masks in the di matrix format
    #     vois_in_dij_format = get_vois_dij_format(tplan_obj,scaled_dij_matrix)
    #
    #     # Add vois in dij format to tpopt_objectives
    #     for count, voi in enumerate(TPopt_objectives['vois']):
    #         dij_format_vois = vois_in_dij_format[f"{voi['name']}"]['1d_mask']
    #         TPopt_objectives['vois'][count]['1d_mask'] = dij_format_vois
    #
    #     # Optimize for plan objective
    #     optimized_fluence, final_iteration_data, iteration_history = optimize_photons_with_scipy(dij_matrix=dij_matrix,fluence_matrix=uniform_fluence_vector,objectives=TPopt_objectives )
    #
    #     # Get dose for fluences
    #     dose = get_dose_from_dijs_and_fluence_vector(dij_matrix, optimized_fluence)
    #
    #     # Plot optimization hisotry
    #     # plot_opti_history(opti_logs)
    #
    #     # Plot dose on ct
    #     plot_dij_matrix_on_ct(dose, plot_note=f'Optimized fluence, Nbeam angles: {nbeams}', voinames_colors_visualization=[('tumor', 'red'),('esophagus','green'),('spinal cord','orange')])

    # #=====================================Week10===============================
    # Implement a proton IMPT optimization for 3 beams.

    # Typical proton plan has 3 angles each at 45 deg diff
    angles = [(360-45),0,(0+45)]

    # use these angles to define the proton beams
    beams = create_photon_beams(angles)
    beams = create_proton_beams(angles)


    # Create dose influence matrix for all beams and beamlets
    dij_matrix, uniform_fluence_vector = create_dij_matrix(beams)
















