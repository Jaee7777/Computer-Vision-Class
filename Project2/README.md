# Project 2
- Student: Jaee Oh

- the command follows following format, assuming Cmake was used on build/ directory:
```
./build/readfiles <directory path> <target file path> [Top N matches] [Matching method] [optional: csv file path] [optional: --least-similar]
```
- [optional: csv file path] field is only required when the moethod requires csv file.
- [optional: --least-similar] can be used for any method.
- Command examples:
```
  ./build/readfiles images/ target.jpg 5 histogram
  ./build/readfiles images/ target.jpg 5 histogram --least-similar
  ./build/readfiles images/ target.jpg 5 dnn-color-texture features.csv
  ./build/readfiles images/ target.jpg 5 dnn-color-texture features.csv --least-similar
```

- Possible Matching methods are:
  - ssd
  - histogram
  - multi-histogram
  - color-texture
  - gabor
  - color-gabor
  - csv-cosine
  - dnn-color-texture 

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

## Part 7: pic.1087.jpg

- DNN + Color + Texture for similar images:
```
=== Top 5 Matches ===
Match 1: data//pic.1086.jpg
  Combined score: 0.8341
Match 2: data//pic.1085.jpg
  Combined score: 0.8324
Match 3: data//pic.1084.jpg
  Combined score: 0.8244
Match 4: data//pic.0711.jpg
  Combined score: 0.7933
Match 5: data//pic.1094.jpg
  Combined score: 0.7763
```

- DNN + Color + Texture for least similar images:
```
=== Top 5 LEAST Similar Matches ===
Match 1: data//pic.0511.jpg
  Combined score: 0.1847
Match 2: data//pic.0084.jpg
  Combined score: 0.2053
Match 3: data//pic.0037.jpg
  Combined score: 0.2545
Match 4: data//pic.0174.jpg
  Combined score: 0.2662
Match 5: data//pic.0377.jpg
  Combined score: 0.2674
```

## Part 7: pic.0516.jpg
- DNN + Color + Texture for similar images:
```
== Top 5 MOST Similar Matches ===
Match 1: data//pic.0526.jpg
  Combined score: 0.7497
Match 2: data//pic.0514.jpg
  Combined score: 0.7359
Match 3: data//pic.0515.jpg
  Combined score: 0.6896
Match 4: data//pic.0151.jpg
  Combined score: 0.6734
Match 5: data//pic.1053.jpg
  Combined score: 0.6728
```

- DNN + Color + Texture for least similar images:
```
=== Top 5 LEAST Similar Matches ===
Match 1: data//pic.1068.jpg
  Combined score: 0.2342
Match 2: data//pic.0251.jpg
  Combined score: 0.2508
Match 3: data//pic.0954.jpg
  Combined score: 0.2546
Match 4: data//pic.1009.jpg
  Combined score: 0.2558
Match 5: data//pic.0250.jpg
  Combined score: 0.2692
```

## Extension 1: pic.1087.jpg
- Dominant color texture using k=5 dominant colors of Kmeans and 16 bins 2D histogram of magnitude Sobel:
```
Top 4 MOST similar matches:
Match 1: data//pic.1087.jpg with score = 0.9989
Match 2: data//pic.0818.jpg with score = 0.9582
Match 3: data//pic.0716.jpg with score = 0.9547
Match 4: data//pic.1084.jpg with score = 0.9462
```

- Edge density using Canny Edge Detection (https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html):
```
Top 4 MOST similar matches:
Match 1: data//pic.1087.jpg with similarity = 1.0000
Match 2: data//pic.0139.jpg with similarity = 0.8912
Match 3: data//pic.0818.jpg with similarity = 0.8908
Match 4: data//pic.0141.jpg with similarity = 0.8867
```

- Chi-square distance on 8-bin 3D RGB channels histogram:
```
Top 4 MOST similar matches:
Match 1: data//pic.1087.jpg with similarity = 1.0000
Match 2: data//pic.1085.jpg with similarity = 0.8996
Match 3: data//pic.1084.jpg with similarity = 0.8782
Match 4: data//pic.1086.jpg with similarity = 0.8575
```

- Hamming distance on 7x7 center feature vector:
```
Top 4 MOST similar matches:
Match 1: data//pic.1087.jpg with similarity = 1.0000
Match 2: data//pic.0348.jpg with similarity = 0.7279
Match 3: data//pic.0491.jpg with similarity = 0.7279
Match 4: data//pic.0478.jpg with similarity = 0.7143
```

## Extension 2: pic.1047.jpg
- dnn-color-texture method applied on images with faces:
```
=== Top 5 MOST Similar Matches (with faces) ===
Match 1: data//pic.0226.jpg
  Combined score: 0.7275
Match 2: data//pic.0321.jpg
  Combined score: 0.7184
Match 3: data//pic.0403.jpg
  Combined score: 0.7121
Match 4: data//pic.0842.jpg
  Combined score: 0.7084
Match 5: data//pic.0006.jpg
  Combined score: 0.7064
```