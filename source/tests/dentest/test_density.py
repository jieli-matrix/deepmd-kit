import pytest
import numpy as np
from density import DensityCalculator
import os

class TestDensityCalculatorRhog:
    @pytest.fixture
    def sample_files(self):
        test_sys = os.path.join(os.path.dirname(__file__), "dataset/Mg1Al20Cu11") # frame 0 in Mg1Al20Cu11
        cube_file = os.path.join(test_sys, "OUT.ABACUS", "SPIN1_CHG.cube")
        binary_file = os.path.join(test_sys, "OUT.ABACUS", "ABACUS-CHARGE-DENSITY.restart")
        return str(cube_file), str(binary_file)

    @pytest.fixture
    def calculator(self, sample_files):
        _, binary_file = sample_files 
        return DensityCalculator(binary_file)

    @pytest.fixture
    def cube_data(self, sample_files):
        cube_file, _ = sample_files
        return self.read_cube_file(cube_file)

    def read_cube_file(self, filename: str) -> dict:
        cube_data = {}
        with open(filename, 'r') as f:
            f.readline()  # Skip first comment line
            f.readline()  # Skip second comment line
            
            parts = f.readline().split()
            cube_data['natoms'] = int(parts[0])
            cube_data['origin'] = np.array([float(p) for p in parts[1:4]])
            
            cube_data['nx'], cube_data['ny'], cube_data['nz'] = [], [], []
            cube_data['grid_spacing'] = np.zeros((3, 3))
            for i in range(3):
                parts = f.readline().split()
                cube_data[f'n{["x", "y", "z"][i]}'] = int(parts[0])
                cube_data['grid_spacing'][i] = [float(p) for p in parts[1:4]]
            
            cube_data['atoms'] = []
            for _ in range(cube_data['natoms']):
                parts = f.readline().split()
                atom_data = [int(parts[0])] + [float(p) for p in parts[1:5]]
                cube_data['atoms'].append(atom_data)
            
            density = []
            for line in f:
                density.extend([float(x) for x in line.split()])
            
            cube_data['density'] = np.array(density).reshape(
                (cube_data['nx'], cube_data['ny'], cube_data['nz'])
            )

        return cube_data

    def test_file_reading(self, calculator):
        assert calculator.ngm_g > 0
        assert calculator.rhog.shape == (calculator.ngm_g,)
        assert calculator.lat_vec.shape == (3, 3)
        assert calculator.g_vectors.shape == (calculator.ngm_g, 3)

    def test_density_calculation(self, calculator, cube_data):
        num_test_points = 1000  # Adjust this number based on your memory constraints
        nx, ny, nz = cube_data['nx'], cube_data['ny'], cube_data['nz']
        
        # Generate random indices
        random_indices = np.random.choice(nx * ny * nz, num_test_points, replace=False)
        x_indices = random_indices // (ny * nz)
        y_indices = (random_indices % (ny * nz)) // nz
        z_indices = random_indices % nz
        
        # Get cube densities at these points
        cube_densities = cube_data['density'][x_indices, y_indices, z_indices]
        
        # Calculate real space coordinates
        grid_coords = np.column_stack([x_indices, y_indices, z_indices])
        real_coords = np.dot(grid_coords, cube_data['grid_spacing']) + cube_data['origin']
    
        
        # Calculate densities using rhog method
        # Note: input points are dimensonless!
        lattice_constant = 1.8897261246257702
        rhog_densities = calculator.calculate_density_batch(real_coords / lattice_constant)
        
        # Compare densities
        relative_diff = np.abs((rhog_densities - cube_densities) / cube_densities)
        max_relative_diff = np.max(relative_diff)
        mean_relative_diff = np.mean(relative_diff)

        print(f"Max relative difference: {max_relative_diff:.5e}")
        print(f"Mean relative difference: {mean_relative_diff:.5e}")

        assert max_relative_diff < 1e-3
        assert mean_relative_diff < 1e-4

    def test_periodic_boundary_conditions(self, calculator):
        lattice_vectors = calculator.get_lattice_vectors()
        point1 = np.array([[0, 0, 0]])
        point2 = lattice_vectors @ np.array([3, 1, 1])
        # Note: input points are dimensonless!
        density1 = calculator.calculate_density_batch(point1)
        density2 = calculator.calculate_density_batch(point2)
        assert np.isclose(density1, density2)
    
    def test_density_grad_calculation(self, calculator, cube_data):
        num_test_points = 1000
        nx, ny, nz = cube_data['nx'], cube_data['ny'], cube_data['nz']
        
        # Generate random indices
        random_indices = np.random.choice(nx * ny * nz, num_test_points, replace=False)
        x_indices = random_indices // (ny * nz)
        y_indices = (random_indices % (ny * nz)) // nz
        z_indices = random_indices % nz
        
        # Get cube densities at these points
        cube_densities = cube_data['density'][x_indices, y_indices, z_indices]
        
        # Calculate real space coordinates
        grid_coords = np.column_stack([x_indices, y_indices, z_indices])
        real_coords = np.dot(grid_coords, cube_data['grid_spacing']) + cube_data['origin']
        
        # Calculate density and gradient
        # Note: input points are dimensonless!
        lattice_constant = 1.8897261246257702
        densities, gradients = calculator.calculate_density_grad_batch(real_coords / lattice_constant)
        
        # Check shapes
        assert densities.shape == (num_test_points,)
        assert gradients.shape == (num_test_points, 3)
        
        # Verify gradient calculation using finite difference
        h = 1e-5  # Small displacement for finite difference
        for i in range(3):
            displacement = np.zeros(3)
            displacement[i] = h
            forward_points = real_coords + displacement
            backward_points = real_coords - displacement
            
            forward_densities = calculator.calculate_density_batch(forward_points / lattice_constant)
            backward_densities = calculator.calculate_density_batch(backward_points / lattice_constant)
            
            numerical_gradient = lattice_constant * (forward_densities - backward_densities) / (2 * h)
            
            assert np.allclose(gradients[:, i], numerical_gradient, rtol=1e-6, atol=1e-6)
        