# Deep-ML Solutions

Personal Deep-ML practice by [Lawson Dong](https://github.com/Lawson-Dong), documenting machine learning implementations, mathematical reasoning, and learning reflections.

## Progress

| Easy | Medium | Hard | Total |
|---|---|---|---|
| 0 | 1 | 0 | 1 |

## Problem Index

| # | Problem | Difficulty | Topics | Approach | Time | Extra Space | Notes | Code |
|---|---|---|---|---|---|---|---|---|
| 015 | [Linear Regression Using Gradient Descent](https://www.deep-ml.com/problems/15) | Medium* | Machine Learning, Linear Regression, PyTorch, Autograd | Full-batch SGD with autograd | O(Tmn) | O(m + n) | [README](medium/015/README.md) | [Python](medium/015/solution_v1.py) |

Here, `T` is the number of iterations, `m` is the number of samples, and `n` is the number of input features, including the intercept column. Space excludes input storage and any tensor conversion copies.

*Difficulty follows the supplied problem screenshot. The current public page labels this problem Easy; platform labels and language variants may differ.*

## Repository Structure

- `medium/015/README.md`: problem description, example, solution explanation, and personal reflection.
- `medium/015/solution_v1.py`: the author's provided PyTorch approach, with minor cleanup.
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
