# Contributing to PiDog Embodiment

Thanks for your interest! We welcome all contributions.

## How to Contribute

### Adding a New Body Adapter

1. Create a new file in `body/adapters/`
2. Implement the `BodyAdapter` interface from `body/adapters/base.py`
3. Add it to `body/adapters/__init__.py`
4. Test with the bridge
5. Submit a PR with documentation

### Adding New Features

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Test thoroughly
5. Submit a PR

### Reporting Issues

- Use GitHub Issues
- Include: hardware, OS version, Python version, error logs
- Screenshots/videos of physical robot behavior are super helpful!

## Code Style

- Python 3.9+ compatible
- Type hints encouraged
- Docstrings for public methods
- Keep it readable — this project is for everyone

## Architecture Decisions

- Brain and body communicate only via HTTP (no shared state)
- Body adapters are the only hardware-specific code
- Security is not optional (always validate, always auth)
- Works offline — LLM is optional, basic functions always work
