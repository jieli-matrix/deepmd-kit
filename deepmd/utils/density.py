import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional
import os
import numpy as np

class DensityCalculator:
    def __init__(self, cube_file: str, binary_file: str, lattice_constant: float):
        self.lattice_constant = lattice_constant  # in Bohr
        self.cube_data = self.read_cube_file(cube_file)
        self.read_binary_file(binary_file)
        self.setup_interpolation()

    def read_cube_file(self, filename: str) -> dict:
        cube_data = {}
        with open(filename, 'r') as f:
            # Skip the first two comment lines
            f.readline()
            f.readline()
            
            # Read number of atoms and origin
            parts = f.readline().split()
            cube_data['natoms'] = int(parts[0])
            cube_data['origin'] = np.array([float(p) for p in parts[1:4]])
            
            # Read grid size and grid spacing vectors
            cube_data['nx'], cube_data['ny'], cube_data['nz'] = [], [], []
            cube_data['grid_spacing'] = np.zeros((3, 3))
            for i in range(3):
                parts = f.readline().split()
                cube_data[f'n{["x", "y", "z"][i]}'] = int(parts[0])
                cube_data['grid_spacing'][i] = [float(p) for p in parts[1:4]]
            
            # Skip atom positions
            # TODO: implement reading atom positions when needed
            for _ in range(cube_data['natoms']):
                f.readline()
            
            # Read density data
            density = []
            for line in f:
                density.extend([float(x) for x in line.split()])
            
            cube_data['density'] = np.array(density).reshape(
                (cube_data['nx'], cube_data['ny'], cube_data['nz'])
            )
            
            # Generate grid
            x = np.linspace(0, 1, cube_data['nx'], endpoint=False)
            y = np.linspace(0, 1, cube_data['ny'], endpoint=False)
            z = np.linspace(0, 1, cube_data['nz'], endpoint=False)
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            cube_data['grid'] = np.dot(np.stack([xx, yy, zz], axis=-1), cube_data['grid_spacing']) + cube_data['origin']

        return cube_data

    def read_binary_file(self, filename: str):
        # 读取二进制文件，获取 g_vectors 和 rhog
        with open(filename, 'rb') as f:
            # Read header
            size = np.fromfile(f, dtype=np.int32, count=1)[0]
            assert size == 3, f"Unexpected header value: {size}"

            self.gammaonly, self.ngm_g, self.nspin = np.fromfile(f, dtype=np.int32, count=3)
            _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the last '3'

            # Read reciprocal lattice vectors
            size = np.fromfile(f, dtype=np.int32, count=1)[0]
            assert size == 9, f"Unexpected header value: {size}"
            self.bmat = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
            _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the last '9'

            # Read Miller indices
            size = np.fromfile(f, dtype=np.int32, count=1)[0]
            assert size == self.ngm_g * 3, f"Unexpected Miller indices size: {size}"
            self.miller_indices = np.fromfile(f, dtype=np.int32, count=self.ngm_g*3).reshape(self.ngm_g, 3)
            _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the size at the end

            # Read rhog
            size = np.fromfile(f, dtype=np.int32, count=1)[0]
            assert size == self.ngm_g, f"Unexpected rhog size: {size}"
            self.rhog = np.fromfile(f, dtype=np.complex128, count=self.ngm_g)
            _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the size at the end

            # If nspin == 2, read second spin component
            if self.nspin == 2:
                size = np.fromfile(f, dtype=np.int32, count=1)[0]
                assert size == self.ngm_g, f"Unexpected rhog_spin2 size: {size}"
                self.rhog_spin2 = np.fromfile(f, dtype=np.complex128, count=self.ngm_g)
                _ = np.fromfile(f, dtype=np.int32, count=1)[0]  # Skip the size at the end

        # Calculate real space lattice vectors (dimensionless)
        self.lat_vec = np.linalg.inv(self.bmat).T

        # Calculate cell volume
        self.cell_volume = self.calculate_cell_volume_from_reciprocal(self.bmat)

        # Calculate G vectors (in units of 2π/lattice_constant)
        self.g_vectors = self.calculate_g_vectors(self.miller_indices, self.bmat)

    @staticmethod
    def calculate_cell_volume_from_reciprocal(reciprocal_lattice: np.ndarray) -> float:
        """
        Calculate the real space cell volume from the reciprocal lattice vectors.
        """
        reciprocal_volume = np.abs(np.linalg.det(reciprocal_lattice))
        real_space_volume = (2 * np.pi) ** 3 / reciprocal_volume
        return real_space_volume

    @staticmethod
    def calculate_g_vectors(miller_indices: np.ndarray, reciprocal_lattice: np.ndarray) -> np.ndarray:
        """
        Calculate the G-vectors from Miller indices and reciprocal lattice vectors.
        The resulting G-vectors are in units of 2π/lattice_constant.
        """
        return np.dot(miller_indices, reciprocal_lattice.T)

    def setup_interpolation(self):
        x = np.linspace(0, 1, self.cube_data['nx'], endpoint=False)
        y = np.linspace(0, 1, self.cube_data['ny'], endpoint=False)
        z = np.linspace(0, 1, self.cube_data['nz'], endpoint=False)
        self.interpolator = RegularGridInterpolator((x, y, z), self.cube_data['density'])

    def calculate_density_interpolated(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate density using interpolation from CUBE file data.
        
        Args:
            points (np.ndarray): Array of points in real space, shape (n, 3), in units of lattice_constant
        
        Returns:
            np.ndarray: Interpolated density values
        """
        frac_coords = self.to_fractional_cube(points / self.lattice_constant)
        return self.interpolator(frac_coords % 1)
    
    def calculate_density_exact(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate the exact density using Fourier transform.
        
        Args:
            points (np.ndarray): Array of points in real space, shape (n, 3), in units of lattice_constant
        
        Returns:
            np.ndarray: Exact density values
        """
        frac_coords = self.to_fractional_exact(points / self.lattice_constant)
        phases = np.exp(1j * 2 * np.pi * np.dot(frac_coords, self.g_vectors.T))
        return np.real(np.dot(phases, self.rhog)) / self.cell_volume

    def to_fractional_cube(self, points: np.ndarray) -> np.ndarray:
        """
        Convert points from Cartesian to fractional coordinates based on CUBE file grid.
        
        Args:
            points (np.ndarray): Array of points in Cartesian coordinates, shape (n, 3)
        
        Returns:
            np.ndarray: Array of points in fractional coordinates, shape (n, 3)
        """
        return np.dot(points - self.cube_data['origin'], np.linalg.inv(self.cube_data['grid_spacing']))
    
    def to_fractional_exact(self, points: np.ndarray) -> np.ndarray:
        """
        Convert points from Cartesian to fractional coordinates based on binary file lattice.
        
        Args:
            points (np.ndarray): Array of points in Cartesian coordinates, shape (n, 3)
        
        Returns:
            np.ndarray: Array of points in fractional coordinates, shape (n, 3)
        """
        return np.dot(points, np.linalg.inv(self.lat_vec))
if __name__ == "__main__":
    test_data_dir = "/root/abacusstru/ternary/Mg1Al20Cu11/0"
    test_cube_f = os.path.join(test_data_dir, "OUT.ABACUS", "SPIN1_CHG.cube")
    test_bin_f = os.path.join(test_data_dir, "OUT.ABACUS", "ABACUS-CHARGE-DENSITY.restart")
    lattice_constant = 1.8897261246257702  # Example value in Bohr (1 Angstrom)
    testcal = DensityCalculator(test_cube_f, test_bin_f, lattice_constant)
