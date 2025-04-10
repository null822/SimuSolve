
# SimuSolve

A GPU-Accelerated Simultaneous Equations Solver, written in C#, using OpenCL. This was designed for a Software Engineering assignment.

## Usage

Run SimuSolve with the following command:

```POWERSHELL
./SimuSolve.exe "path/to/input.csv" "path/to/output.csv"
```

When given no arguments, SimuSolve defaults the files `data/input.csv` and `data/output.csv`.
### CSV Layout

Example Simultaneous Equations:

$Z_{1} = a_{1}A + b_{1}B + c_{1}C$ \
$Z_{2} = a_{2}A + b_{2}B + c_{2}C$ \
$Z_{3} = a_{3}A + b_{3}B + c_{3}C$

#### Input File

The first column of the input CSV file stores the constant values (left of the equals sign), and all remaining columns store the coefficients. Each row represents an equation.

#### Output File

The output CSV file stores the solutions, where the first entry is the solution for the first coefficient, and the last entry is for the last coefficient.
