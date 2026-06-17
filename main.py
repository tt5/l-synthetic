#!/usr/bin/env python3
"""generate.py

Generate synthetic shape images using lines and segments.

Usage:
    uv run generate.py
"""


class Segment:
    """A single straight segment of a line.
    
    Attributes:
        length: int — length of the segment in pixels
        angle: int — angle in degrees (relative to previous segment)
        line: Line — the line this segment belongs to
    """

    def __init__(self, length, angle, line):
        self.length = length
        self.angle = angle
        self.line = line


class Line:
    """A line composed of connected segments.
    
    Attributes:
        segments: list of Segment — segments making up this line
        image: Image — the image this line belongs to
        line_id: int — unique identifier for this line
    """

    def __init__(self, image, line_id):
        self.segments = []
        self.image = image
        self.line_id = line_id

    def add_segment(self, segment):
        """Add a segment to this line.
        
        Returns:
            bool — True if segment was added successfully, False otherwise
        """
        # TODO: Implement
        pass


class Image:
    """A 27x27 black/white image containing lines.
    
    Attributes:
        num_lines: int — number of lines in this image
        thickness: int — line thickness (same for all lines)
        lines: list of Line — lines in this image
    """

    def __init__(self, num_lines, thickness):
        self.num_lines = num_lines
        self.thickness = thickness
        self.lines = []

    def add_line(self, line):
        """Add a line to this image.
        
        Does not apply composition algorithm — just adds the line.
        """
        # TODO: Implement
        pass


def main():
    print("Synthetic pictogram generator")
    print("TODO: Implement generation algorithm")


if __name__ == "__main__":
    main()
