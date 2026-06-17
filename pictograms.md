# pictograms

An image has at least 1 line. Lines have segments.

## line

- length
- number of segments

## segments

- length
- angle

## image

- 27x27, black/white
- number of lines
- every line is on its own layer

## adding lines to image

1. start with empty image (all white, zero lines)
2. add one more
3. subtract old projected image from new projected image, if there are more than 1 line chose only one of them.
4. repeat until we reach the number of lines in the image
