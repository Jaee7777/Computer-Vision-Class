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

## Part 1: SSD
```
Top 5 matches:
Match 1: data/pic.1016.jpg with SSD = 0.00
Match 2: data/pic.0986.jpg with SSD = 14049.00
Match 3: data/pic.0641.jpg with SSD = 21756.00
Match 4: data/pic.0547.jpg with SSD = 49703.00
Match 5: data/pic.1013.jpg with SSD = 51539.00
```

## Part 2: Histogram
```
Top 5 matches:
Match 1: data/pic.0164.jpg with intersection = 1.0000
Match 2: data/pic.0110.jpg with intersection = 0.4246
Match 3: data/pic.1032.jpg with intersection = 0.4234
Match 4: data/pic.0092.jpg with intersection = 0.3891
Match 5: data/pic.0976.jpg with intersection = 0.3734
```

## Part 3: Multi-histogram
```
Top 5 matches:
Match 1: data/pic.0274.jpg with intersection = 1.0000
Match 2: data/pic.0273.jpg with intersection = 0.6527
Match 3: data/pic.1031.jpg with intersection = 0.6249
Match 4: data/pic.0409.jpg with intersection = 0.6205
Match 5: data/pic.0213.jpg with intersection = 0.5911
```

## Part 4: Texture methods

- color-texture: Equally weighted 8 bins of RGB channels and 16 bins of gradient magnitude and orientation using Sobel
```
Top 4 matches:
Match 1: data/pic.0535.jpg with combined score = 1.0000
Match 2: data/pic.0171.jpg with combined score = 0.7965
Match 3: data/pic.0628.jpg with combined score = 0.7906
Match 4: data/pic.0454.jpg with combined score = 0.7751
```

- gabor: Gabor filter of k=21, sigma=5.0, gamm=0.5, psi=0 using cosine similarity
```
Top 4 matches:
Match 1: data/pic.0535.jpg with Gabor similarity = 1.0000
Match 2: data/pic.0298.jpg with Gabor similarity = 0.9998
Match 3: data/pic.0819.jpg with Gabor similarity = 0.9997
Match 4: data/pic.0449.jpg with Gabor similarity = 0.9997
```

- color-gabor: equally weighted 8 bins for RGB channels and Gabor filter of k=21, sigma=5.0, gamm=0.5, psi=0
```
Top 4 matches:
Match 1: data/pic.0535.jpg with combined score = 1.0000
Match 2: data/pic.0285.jpg with combined score = 0.8877
Match 3: data/pic.0628.jpg with combined score = 0.8865
Match 4: data/pic.0337.jpg with combined score = 0.8681
```

## Part 5: Deep network embedding (csv-cosine)
- csv-cosine with target image pic.0893.jpg
```
Top 3 matches:
Match 1: pic.0897.jpg with cosine distance = 0.1518
Match 2: pic.0136.jpg with cosine distance = 0.1762
Match 3: pic.0146.jpg with cosine distance = 0.2249
```

- csv-cosine with target image pic.0164.jpg
```
Top 3 matches:
Match 1: pic.1032.jpg with cosine distance = 0.2122
Match 2: pic.0213.jpg with cosine distance = 0.2128
Match 3: pic.0690.jpg with cosine distance = 0.2351
```

## Part 6: Comparison for pic.1072.jpg
- csv-cosine
```
Top 3 matches:
Match 1: pic.0143.jpg with cosine distance = 0.1610
Match 2: pic.0863.jpg with cosine distance = 0.2004
Match 3: pic.0329.jpg with cosine distance = 0.2072
```
- gabor
```
Top 4 matches:
Match 1: data/pic.1072.jpg with Gabor similarity = 1.0000
Match 2: data/pic.0700.jpg with Gabor similarity = 0.9998
Match 3: data/pic.0239.jpg with Gabor similarity = 0.9998
Match 4: data/pic.0272.jpg with Gabor similarity = 0.9997
```

- color-texture
```
Top 4 matches:
Match 1: data/pic.1072.jpg with combined score = 1.0000
Match 2: data/pic.0701.jpg with combined score = 0.7987
Match 3: data/pic.0951.jpg with combined score = 0.7902
Match 4: data/pic.0937.jpg with combined score = 0.7884
```

- multi-histogram
```
Top 4 matches:
Match 1: data/pic.1072.jpg with intersection = 1.0000
Match 2: data/pic.0813.jpg with intersection = 0.6468
Match 3: data/pic.0701.jpg with intersection = 0.6461
Match 4: data/pic.1069.jpg with intersection = 0.6387
```

- histogram
```
Top 4 matches:
Match 1: data/pic.1072.jpg with intersection = 1.0000
Match 2: data/pic.0937.jpg with intersection = 0.7072
Match 3: data/pic.0732.jpg with intersection = 0.6853
Match 4: data/pic.0673.jpg with intersection = 0.6610
```

- ssd
```
Top 4 matches:
Match 1: data/pic.1072.jpg with SSD = 0.00
Match 2: data/pic.0768.jpg with SSD = 78740.00
Match 3: data/pic.0138.jpg with SSD = 79295.00
Match 4: data/pic.0303.jpg with SSD = 82805.00
```

## Part 6: Comparison for pic.0948.jpg
- csv-cosine
```
Top 3 matches:
Match 1: pic.0930.jpg with cosine distance = 0.1283
Match 2: pic.0960.jpg with cosine distance = 0.2004
Match 3: pic.0928.jpg with cosine distance = 0.2041
```

- gabor
```
Top 4 matches:
Match 1: data/pic.0948.jpg with Gabor similarity = 1.0000
Match 2: data/pic.0752.jpg with Gabor similarity = 0.9999
Match 3: data/pic.0767.jpg with Gabor similarity = 0.9999
Match 4: data/pic.0130.jpg with Gabor similarity = 0.9999
```

- color-texture
```
Top 4 matches:
Match 1: data/pic.0948.jpg with combined score = 1.0000
Match 2: data/pic.0891.jpg with combined score = 0.7859
Match 3: data/pic.0675.jpg with combined score = 0.7839
Match 4: data/pic.1069.jpg with combined score = 0.7831
```

- multi-histogram
```
Top 4 matches:
Match 1: data/pic.0948.jpg with intersection = 1.0000
Match 2: data/pic.0217.jpg with intersection = 0.7323
Match 3: data/pic.0675.jpg with intersection = 0.7150
Match 4: data/pic.0696.jpg with intersection = 0.7032
```

- histogram
```
Top 4 matches:
Match 1: data/pic.0948.jpg with intersection = 1.0000
Match 2: data/pic.0217.jpg with intersection = 0.7504
Match 3: data/pic.0675.jpg with intersection = 0.7340
Match 4: data/pic.0735.jpg with intersection = 0.7311
```

- ssd
```
Top 4 matches:
Match 1: data/pic.0948.jpg with SSD = 0.00
Match 2: data/pic.0176.jpg with SSD = 1680.00
Match 3: data/pic.0668.jpg with SSD = 1810.00
Match 4: data/pic.0064.jpg with SSD = 1954.00
```