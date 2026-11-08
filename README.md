# Deep-ML Solutions

Personal Deep-ML practice by [Lawson Dong](https://github.com/Lawson-Dong), documenting machine learning implementations, mathematical reasoning, and learning reflections.

## Progress

| Easy | Medium | Hard | Total |
|---|---|---|---|
| 1 | 3 | 0 | 4 |

## Problem Index

| # | Problem | Difficulty | Topics | Approach | Time | Extra Space | Notes | Code |
|---|---|---|---|---|---|---|---|---|
| 001 | [Matrix-Vector Dot Product](https://www.deep-ml.com/problems/1) | Easy | Linear Algebra, PyTorch | Matrix–vector multiplication with `@` | O(mn) | O(m) | [README](easy/001/README.md) | [Python](easy/001/solution_v1.py) |
| 014 | [Linear Regression Using Normal Equation](https://www.deep-ml.com/problems/14) | Medium | Machine Learning, Linear Regression, PyTorch, Least Squares | Direct least squares with `torch.linalg.lstsq` | O(mn²) | O(mn) | [README](medium/014/README.md) | [Python](medium/014/solution_v1.py) |
| 015 | [Linear Regression Using Gradient Descent](https://www.deep-ml.com/problems/15) | Medium* | Machine Learning, Linear Regression, PyTorch, Autograd | Full-batch SGD with autograd | O(Tmn) | O(m + n) | [README](medium/015/README.md) | [Python](medium/015/solution_v1.py) |
| 1223 | [Gradient of a Weighted Sum of Squares](https://www.deep-ml.com/problems/1223) | Medium | PyTorch, Autograd, Element-wise Operations | Autograd on a scalar sum of squares | O(n) | O(n) | [README](medium/1223/README.md) | [Python](medium/1223/solution_v1.py) |

Here, `T` is the number of iterations, `m` is the number of samples, and `n` is the number of input features, including the intercept column. For problem 001, `m` and `n` are the matrix row and column counts. Space excludes input storage and any tensor conversion copies. Problem 014's complexity bounds assume `m >= n`. For problem 1223, `n` is the input vector length.

*Difficulty follows the supplied problem screenshot. The current public page labels this problem Easy; platform labels and language variants may differ.*

## Repository Structure

- `easy/001/README.md`: matrix–vector product problem, example, solution, and reflection.
- `easy/001/solution_v1.py`: the author's PyTorch matrix–vector multiplication implementation.
- `medium/014/README.md`: normal-equation problem, example, solution explanation, and personal reflection.
- `medium/014/solution_v1.py`: the author's PyTorch least-squares implementation.
- `medium/015/README.md`: problem description, example, solution explanation, and personal reflection.
- `medium/015/solution_v1.py`: the author's provided PyTorch approach, with minor cleanup.
- `medium/1223/README.md`: weighted sum of squares problem, example, solution, and reflection.
- `medium/1223/solution_v1.py`: the author's autograd implementation.
- `requirements.txt`: runtime dependencies.

Problems are grouped by difficulty and use at least three digits for their problem numbers, following the structure of [leetcode-solution](https://github.com/Lawson-Dong/leetcode-solution).

## Documentation Conventions

Each problem README includes:

1. Problem description and requirements.
2. Example inputs, expected outputs, and reasoning.
3. Solution, implementation link, and complexity.
4. Personal reflection supplied by the author and edited for clarity.

All documentation and code comments are in English. The first version preserves the author's supplied approach. Additional versions are added only for distinct approaches or meaningful improvements.

## Running a Solution

Install dependencies with `python -m pip install -r requirements.txt`, then import or run the relevant solution using the example in its README.

Check the selected platform language variant before submitting: the supplied PyTorch solution returns a Tensor, while the generic NumPy statement asks for a NumPy array.

## Maintenance

1. Create `<difficulty>/<problem-number>/`.
2. Add a README with the four required sections and the solution file.
3. Check the example, output shape, and relevant boundary cases.
4. Update the problem index and progress counts.
5. Keep reflections faithful to the author's experience; leave a placeholder when none is provided.
