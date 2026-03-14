# Contributing to CorridorKey

Thank you for your interest in contributing to CorridorKey! We welcome contributions from everyone.

## Before You Start

Please review our [Developer Setup Guide](docs/dev/developer-setup.md) for environment configuration and initial setup.

## Making Changes

1. **Clone the Repository**
   This is the primary development repository.

   ```bash
   git clone https://github.com/nikopueringer/CorridorKey.git
   cd CorridorKey
   ```

2. **Create a Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Keep commits atomic** - one logical change per commit
4. **Write clear commit messages** - follow [Conventional Commits](https://www.conventionalcommits.org/)
   - Format: `<type>(<scope>): <subject>`
   - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`
   - Example: `feat(inference): add support for dynamic batch sizes`

## Submitting a Pull Request

1. Push to your fork: `git push origin feature/your-feature-name`
2. Create a PR with a descriptive title
3. Reference any related issues
4. Describe the changes and why they're needed
5. Ensure all tests pass

## Documentation

When adding features, update relevant documentation in `docs/`. See our [Documentation Principles](docs/dev/documentation-principles/) and [Authoring Guide](docs/dev/authoring-documentation.md).

## Reporting Issues

- Check [existing issues](https://github.com/nikopueringer/CorridorKey/issues) first
- Include steps to reproduce and environment details
- Provide error logs if applicable

## Code Standards

See [Developer Documentation](docs/dev/) for coding standards, linting, and type hints.

---

Thank you for contributing to CorridorKey!
