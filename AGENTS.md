# Agents.md

## Dependency Management

- use `uv` to manage dependencies
- To add a dependency, use `uv add` and don't modify the uv.lock or pyproject.toml files manually. 
- To run any of the module level code from the commandline use `uv run`
- To add development specific dependencies use the `--dev` flag

## Code Style

- Use Ruff for code linting and formatting

## Repostiory Expectations

-- Before making changes to the repository always show a Patch for the proposed code changes and ask for approval. 

## Documentation

- Use Quarto for producting documentation 
- Prefer a slide format
- Save docs in the docs/ folder