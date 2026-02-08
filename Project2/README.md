# Project 2
- Student: Jaee Oh

- the command follows following format, assuming Cmake was used on build/ directory:
```
./build/readfiles <directory path> <target file path> [Top N matches] [Matching method] [optional: csv file path]
```
- [optional: csv file path] field is only required when `csv-cosine` method is selected.

- Possible Matching methods are:
  - ssd
  - histogram
  - multi-histogram
  - color-texture
  - gabor
  - color-gabor
  - csv-cosine