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

- Use GitHub Issues — the bug template asks for the two things that diagnose
  90% of robot problems: `./scripts/doctor.sh` output and the service journal
- Screenshots/videos of physical robot behavior are super helpful!

## Running Tests

All tests run on any machine — no robot, no Pi, no network:

```bash
PYTHONPATH=. python3 -m pytest tests/ -q        # full suite (~6 s)
PYTHONPATH=. python3 -m pytest tests/test_mcp_server.py -v   # one file
```

## Developing Without Hardware

```bash
./install.sh --mock          # a pretend dog on 127.0.0.1:8888 (no sudo needed)
# second terminal:
curl http://127.0.0.1:8888/status
python3 examples/basic_control.py
```

The mock answers with the same API shapes as the real bridge — tests pin the
two together, so code that works against the mock works on the robot.

## CI Gates

- `ruff check --select F,E9` **blocks** the build (undefined names, unused
  imports, syntax) — run it locally before pushing
- the full style check and mypy are advisory
- `tests/test_no_private_data.py` fails the build on hardcoded IPs,
  credentials, personal names or home paths — keep config in env files

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
