# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from typing import Tuple
class DensityCalculator:
    def __init__(self, binary_file: str):
        self.read_binary_file(binary_file)

    def read_binary_file(self, filename: str):
        # Read the binary file for g_vectors & rhog
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

        # Calculate G vectors (in units of 2π/lattice_constant)
        self.bmat = 2 * np.pi * self.bmat 
        # self.g_vectors = 2 * np.pi * np.dot(self.miller_indices, self.bmat.T)
        self.g_vectors = np.dot(self.miller_indices, self.bmat.T)
    def calculate_density_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate the exact density using Fourier transform.
        
        Args:
            points (np.ndarray): Array of points in real space, shape (n, 3), in units of lattice_constant
        
        Returns:
            np.ndarray: Exact density values
        """
        phases = np.exp(1j * np.dot(points, self.g_vectors.T))
        densities = np.real(np.dot(phases, self.rhog))
        return densities

    def get_lattice_vectors(self) -> np.ndarray:
        """
        Get the real space lattice vectors.
        
        Returns:
            np.ndarray: Real space lattice vectors, shape (3, 3), in Angstrom
        """
        return self.lat_vec
    

# def generate_grid(
#     lattice_vectors: np.ndarray,
#     grid_size: Tuple[int, int, int],
#     origin: np.ndarray = np.zeros(3),
# ) -> np.ndarray:
#     """
#     生成给定晶格和网格大小的实空间网格点。

#     参数:
#     lattice_vectors (np.ndarray): 形状为 (3, 3) 的晶格向量
#     grid_size (Tuple[int, int, int]): 三个方向上的网格点数
#     origin (np.ndarray): 网格原点坐标，默认为 (0, 0, 0)

#     返回:
#     np.ndarray: 形状为 (N, 3) 的网格点坐标数组
#     """
#     nx, ny, nz = grid_size
#     x = np.linspace(0, 1, nx, endpoint=False)
#     y = np.linspace(0, 1, ny, endpoint=False)
#     z = np.linspace(0, 1, nz, endpoint=False)

#     grid = np.meshgrid(x, y, z, indexing="ij")
#     points = np.stack(grid, axis=-1).reshape(-1, 3)

#     # 将分数坐标转换为实空间坐标
#     real_points = np.dot(points, lattice_vectors) + origin

#     return real_points


# def calculate_density(
#     filename: str,
#     lattice_vectors: np.ndarray,
#     grid_size: Tuple[int, int, int],
#     origin: np.ndarray = np.zeros(3),
#     batch_size: int = 1000,
# ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
#     """
#     计算给定网格的电子密度，以迭代器形式返回结果。

#     参数:
#     filename (str): 包含 n(G) 数据的二进制文件路径
#     grid_size (Tuple[int, int, int]): 三个方向上的网格点数
#     origin (np.ndarray): 网格原点坐标，默认为 (0, 0, 0)
#     lattice_vectors (np.ndarray): 形状为 (3, 3) 的晶格向量
#     batch_size (int): 每批处理的点数

#     返回:
#     Iterator[Tuple[np.ndarray, np.ndarray]]: yielding (坐标, 密度值) 对
#     """
#     calculator = DensityCalculator(filename)

#     points = generate_grid(lattice_vectors, grid_size, origin)

#     for i in range(0, len(points), batch_size):
#         batch_points = points[i : i + batch_size]
#         batch_densities = calculator.calculate_density_batch(batch_points)
#         yield batch_points, batch_densities
