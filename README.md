# Tutor test

A small command-line tool that measures how well and how fast local AI models
(Microsoft Foundry Local) give beginner Python hints on a given computer. It
decides which model the Python learning app's offline tutor should use.

For each model it sets up GPU/NPU acceleration where available, downloads the
model if needed, loads it, warms it up, then asks for a hint on 15 typical
beginner mistakes (`cases.toml`) using the tutor prompt (`src/prompt.rs`). It
records time to first words, total time, tokens per second and some rough
checks (on topic, didn't give the answer away, no code block, under 110 words),
and writes `report.html` + `results.json`.

## Use

See `docs/HOW-TO-RUN.txt` (shipped in the zip). Options: `tutor-test --help`.

Default models: `qwen2.5-coder-1.5b`, `phi-4-mini`. `--with-7b` adds
`qwen2.5-coder-7b`. Models are cached in `~/.tutor-test`.

## Build

The engine (`foundry_local` + ONNX Runtime + GenAI) is fetched from NuGet at
build time when `FOUNDRY_LOCAL_RUNTIME_VERSION` is set, and must sit next to
the program at run time. CI does this and packages the result.

```sh
cargo test                       # unit tests, no engine needed
FOUNDRY_LOCAL_RUNTIME_VERSION=x.y.z cargo build --release
```

CI (`.github/workflows/build.yml`) on Windows x64 and macOS arm64: lint, unit
tests, release build, packages the program with the engine libraries (and the
Visual C++ runtime on Windows), runs an end-to-end test with `qwen2.5-0.5b`,
and uploads `tutor-test-windows-x64` / `tutor-test-mac-arm64` artifacts.

## Layout

- `src/cases.rs` + `cases.toml`: the beginner mistakes
- `src/prompt.rs`: the tutor prompt
- `src/score.rs`: rough automatic checks
- `src/runner.rs`: Foundry Local (accelerators, download, load, timed streaming hints)
- `src/report.rs`, `src/results.rs`: report and results
- `src/system.rs`: computer facts, free disk space
- `src/args.rs`: options
