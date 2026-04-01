# Phase 0 — Project Setup (1 week)

## Goal
Set up CMake project, integrate gtest and protobuf, compile and run a "hello world".

## Tasks

- [ ] Create CMake project (CMakeLists.txt)
- [ ] Directory structure: src/, include/, tests/, examples/
- [ ] Integrate Google Test (gtest) via FetchContent
- [ ] Add protobuf library via FetchContent
- [ ] src/main.cpp — simple main that prints "Engine initialized"
- [ ] Add first test file under tests/
- [ ] Build and run: cmake .. && make && ./edge_ml_engine
- [ ] Add GitHub Actions CI workflow

## Notes

_Update this section as the phase progresses: problems encountered, decisions made, lessons learned._

## Resources

- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/)
- [GoogleTest Quickstart](https://google.github.io/googletest/quickstart-cmake.html)
- [Protobuf C++ Tutorial](https://protobuf.dev/getting-started/cpptutorial/)
