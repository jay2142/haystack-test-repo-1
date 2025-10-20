"""Core Matrix class and operations."""

import math
from typing import List, Union


class Matrix:
    """A matrix class supporting basic matrix operations."""

    def __init__(self, data: List[List[float]]):
        """
        Initialize a matrix from a 2D list.

        Args:
            data: 2D list representing the matrix

        Raises:
            ValueError: If rows have inconsistent lengths
        """
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty - provide at least a 1x1 matrix")

        row_length = len(data[0])
        if not all(len(row) == row_length for row in data):
            row_lengths = [len(row) for row in data]
            raise ValueError(
                f"All rows must have the same length. Got varying lengths: {row_lengths}"
            )

        self.data = [row[:] for row in data]  # Deep copy
        self.rows = len(data)
        self.cols = len(data[0])

    def __repr__(self) -> str:
        """String representation of the matrix."""
        return f"Matrix({self.data})"

    def __str__(self) -> str:
        """Pretty print the matrix."""
        max_width = max(len(f"{val:.2f}") for row in self.data for val in row)
        lines = []
        for row in self.data:
            formatted_row = [f"{val:>{max_width}.2f}" for val in row]
            lines.append("[" + ", ".join(formatted_row) + "]")
        return "\n".join(lines)

    def __eq__(self, other: 'Matrix') -> bool:
        """Check if two matrices are equal."""
        if not isinstance(other, Matrix):
            return False
        return self.data == other.data

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """
        Add two matrices.

        Args:
            other: Matrix to add

        Returns:
            New matrix representing the sum

        Raises:
            ValueError: If matrices have different dimensions
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Matrices must have same dimensions for addition. "
                f"Got {self.rows}x{self.cols} and {other.rows}x{other.cols}"
            )

        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] + other.data[i][j])
            result.append(row)

        return Matrix(result)

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """
        Subtract two matrices.

        Args:
            other: Matrix to subtract

        Returns:
            New matrix representing the difference

        Raises:
            ValueError: If matrices have different dimensions
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Matrices must have same dimensions for subtraction. "
                f"Got {self.rows}x{self.cols} and {other.rows}x{other.cols}"
            )

        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] - other.data[i][j])
            result.append(row)

        return Matrix(result)

    def __mul__(self, other: Union['Matrix', float]) -> 'Matrix':
        """
        Multiply matrix by another matrix or a scalar.

        Args:
            other: Matrix or scalar to multiply by

        Returns:
            New matrix representing the product

        Raises:
            ValueError: If matrix dimensions are incompatible
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    row.append(self.data[i][j] * other)
                result.append(row)
            return Matrix(result)

        elif isinstance(other, Matrix):
            # Matrix multiplication
            if self.cols != other.rows:
                raise ValueError(
                    f"Cannot multiply {self.rows}x{self.cols} matrix "
                    f"by {other.rows}x{other.cols} matrix"
                )

            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    value = sum(
                        self.data[i][k] * other.data[k][j]
                        for k in range(self.cols)
                    )
                    row.append(value)
                result.append(row)

            return Matrix(result)

        else:
            raise TypeError(f"Cannot multiply Matrix by {type(other)}")

    def __rmul__(self, other: float) -> 'Matrix':
        """Right multiplication for scalar * matrix."""
        return self.__mul__(other)

    def get(self, row: int, col: int) -> float:
        """
        Get element at specified position.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)

        Returns:
            Value at the specified position
        """
        return self.data[row][col]

    def set(self, row: int, col: int, value: float) -> None:
        """
        Set element at specified position.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            value: Value to set
        """
        self.data[row][col] = value

    def copy(self) -> 'Matrix':
        """Create a deep copy of the matrix."""
        return Matrix(self.data)
