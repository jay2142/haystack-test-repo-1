"""Core Matrix class implementation."""

from typing import List, Union, Tuple, Optional
import math


class Matrix:
    """
    A flexible matrix class with comprehensive operations.

    Attributes:
        data: 2D list containing matrix elements
        shape: Tuple of (rows, cols)
    """

    def __init__(self, data: List[List[Union[int, float]]]):
        """
        Initialize a matrix.

        Args:
            data: 2D list of numbers

        Raises:
            ValueError: If data is invalid or inconsistent
        """
        if not data:
            raise ValueError("Cannot create matrix from empty data")

        if not all(isinstance(row, list) for row in data):
            raise ValueError("Data must be a list of lists")

        row_len = len(data[0])
        if not all(len(row) == row_len for row in data):
            raise ValueError(
                f"Inconsistent row lengths: expected {row_len}, "
                f"got {[len(row) for row in data]}"
            )

        self.data = [row[:] for row in data]
        self.shape = (len(data), len(data[0]))

    @property
    def rows(self) -> int:
        """Get number of rows."""
        return self.shape[0]

    @property
    def cols(self) -> int:
        """Get number of columns."""
        return self.shape[1]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Matrix({self.shape[0]}x{self.shape[1]})"

    def __str__(self) -> str:
        """Return formatted string for printing."""
        lines = []
        for row in self.data:
            formatted = [f"{val:8.3f}" for val in row]
            lines.append("[ " + " ".join(formatted) + " ]")
        return "\n".join(lines)

    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Get element at position [i, j]."""
        i, j = key
        return self.data[i][j]

    def __setitem__(self, key: Tuple[int, int], value: float) -> None:
        """Set element at position [i, j]."""
        i, j = key
        self.data[i][j] = value

    def __eq__(self, other: 'Matrix') -> bool:
        """Check equality with another matrix."""
        if not isinstance(other, Matrix):
            return False
        return self.data == other.data

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Add two matrices."""
        if self.shape != other.shape:
            raise ValueError(
                f"Shape mismatch: {self.shape} vs {other.shape}"
            )

        result = []
        for i in range(self.rows):
            row = [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            result.append(row)

        return Matrix(result)

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Subtract two matrices."""
        if self.shape != other.shape:
            raise ValueError(
                f"Shape mismatch: {self.shape} vs {other.shape}"
            )

        result = []
        for i in range(self.rows):
            row = [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            result.append(row)

        return Matrix(result)

    def __mul__(self, other: Union['Matrix', float]) -> 'Matrix':
        """Multiply matrix by scalar or another matrix."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result = [[val * other for val in row] for row in self.data]
            return Matrix(result)

        elif isinstance(other, Matrix):
            # Matrix multiplication
            if self.cols != other.rows:
                raise ValueError(
                    f"Cannot multiply {self.shape} by {other.shape}: "
                    f"inner dimensions must match"
                )

            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    val = sum(self.data[i][k] * other.data[k][j]
                             for k in range(self.cols))
                    row.append(val)
                result.append(row)

            return Matrix(result)

        raise TypeError(f"Cannot multiply Matrix by {type(other)}")

    def __rmul__(self, scalar: float) -> 'Matrix':
        """Right multiplication by scalar."""
        return self.__mul__(scalar)

    def transpose(self) -> 'Matrix':
        """Return the transpose of the matrix."""
        result = [[self.data[i][j] for i in range(self.rows)]
                  for j in range(self.cols)]
        return Matrix(result)

    def is_square(self) -> bool:
        """Check if matrix is square."""
        return self.rows == self.cols

    def copy(self) -> 'Matrix':
        """Create a deep copy of the matrix."""
        return Matrix(self.data)

    def get_row(self, index: int) -> List[float]:
        """Get a specific row."""
        if not 0 <= index < self.rows:
            raise IndexError(f"Row index {index} out of range [0, {self.rows})")
        return self.data[index][:]

    def get_col(self, index: int) -> List[float]:
        """Get a specific column."""
        if not 0 <= index < self.cols:
            raise IndexError(f"Column index {index} out of range [0, {self.cols})")
        return [row[index] for row in self.data]

    @staticmethod
    def identity(size: int) -> 'Matrix':
        """Create an identity matrix of given size."""
        data = [[1.0 if i == j else 0.0 for j in range(size)]
                for i in range(size)]
        return Matrix(data)

    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """Create a zero matrix."""
        data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)

    @staticmethod
    def ones(rows: int, cols: int) -> 'Matrix':
        """Create a matrix filled with ones."""
        data = [[1.0 for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)
