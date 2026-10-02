import cv2
import turtle
import numpy as np
import os
import gc
import random
import time

# Initialize Turtle screen configuration
SCREEN_WIDTH = 736
SCREEN_HEIGHT = 736
screen = turtle.Screen()
screen.setup(SCREEN_WIDTH, SCREEN_HEIGHT)
screen.tracer(0)
screen.bgcolor("white")


def quantize_colors(image, k=16):
    """Reduce the number of colors in an image using k-means clustering."""
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()].reshape(image.shape)
    return quantized_image


def outline(image):
    """Load image, blur, and apply adaptive thresholding for edge detection."""
    src_image = cv2.imread(image)
    if src_image is None:
        raise FileNotFoundError(f"Could not load image file: {image}")

    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive thresholding for better edge detection
    th3 = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=7,
        C=1,
    )
    return th3, src_image


def draw_and_fill_areas(image, width, height, x, y):
    """Quantize colors, extract contours, and fill polygon areas using Turtle."""
    fill_turtle = turtle.Turtle()
    fill_turtle.speed(0)
    fill_turtle.hideturtle()

    # Quantize colors for better recognition (lowered k slightly for speed and responsiveness)
    quantized_image = quantize_colors(image, k=24)
    hsv = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2HSV)

    # Find unique colors
    unique_colors = np.unique(hsv.reshape(-1, 3), axis=0)

    count = 0
    for color in unique_colors:
        lower = color
        upper = color + 16

        mask = cv2.inRange(hsv, lower, upper)

        # Dilate to fill small gaps in contours
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Convert HSV color to RGB
        rgb_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0]
        rgb_normalized = tuple(rgb_color.astype(float) / 255)

        for contour in contours:
            if len(contour) > 3:  # Ignore very small shapes
                points = [
                    (
                        p[0][0] - width / 2 + x,
                        -1 * (p[0][1] - height / 2 + y),
                    )
                    for p in contour
                ]

                fill_turtle.penup()
                fill_turtle.goto(points[0])
                fill_turtle.color(rgb_normalized, rgb_normalized)
                fill_turtle.begin_fill()

                for px, py in points:
                    fill_turtle.goto(px, py)

                fill_turtle.end_fill()

                count += 1
                # Periodically update screen and process window events to prevent freezing
                if count % 10 == 0:
                    screen.update()


def find_next_connected_point(current, positions, visited, max_distance=5):
    """Find the next connected point in a continuous line sequence."""
    if not positions:
        return None

    current = np.array(current)
    pos_array = np.array(positions)

    # Calculate distances to all points
    distances = np.sqrt(np.sum((pos_array - current) ** 2, axis=1))

    # Find points within max_distance
    close_points_idx = np.where(distances <= max_distance)[0]
    if len(close_points_idx) == 0:
        return None

    for idx in close_points_idx[np.argsort(distances[close_points_idx])]:
        point = tuple(pos_array[idx])
        if point not in visited:
            return positions[idx]

    return None


def draw_image(image_file, x, y):
    """Main function to handle color filling and continuous outline drawing."""
    try:
        im = cv2.imread(image_file)
        if im is None:
            raise FileNotFoundError(f"Could not load image file: {image_file}")

        th3, color_image = outline(image_file)
        WIDTH = im.shape[1]
        HEIGHT = im.shape[0]

        # Fill colors first
        draw_and_fill_areas(im, WIDTH, HEIGHT, x, y)

        # Process edges for drawing outlines
        iH, iW = np.where(th3 == 0)
        screen_x = iW - WIDTH / 2 + x
        screen_y = -1 * (iH - HEIGHT / 2 + y)
        positions = [list(pos) for pos in zip(screen_x, screen_y)]
        colors = [tuple(color_image[h, w][::-1] / 255) for h, w in zip(iH, iW)]
        position_colors = dict(zip(map(tuple, positions), colors))

        # Single turtle for precise outline drawing
        t = turtle.Turtle()
        t.speed(0)  # Maximum speed to reduce lag
        t.pensize(2)
        t.hideturtle()

        visited = set()
        step_counter = 0

        while positions:
            t.penup()
            start = positions.pop(0)
            t.goto(start)
            t.pendown()

            current = start
            visited.add(tuple(current))

            while current:
                t.pencolor(position_colors[tuple(current)])
                next_point = find_next_connected_point(
                    current, positions, visited
                )
                if next_point:
                    t.pensize(random.uniform(1, 3))
                    t.goto(next_point)
                    visited.add(tuple(next_point))
                    positions.remove(next_point)
                    current = next_point

                    step_counter += 1
                    # Update screen and allow Windows to breathe every 500 steps
                    if step_counter % 500 == 0:
                        screen.update()
                else:
                    current = None

        screen.update()
        print("Drawing finished successfully!")

    except Exception as e:
        print(f"Error in draw_image: {e}")

    finally:
        gc.collect()


# Entry point
image_path = r"c:\Users\ASUS\Downloads\flowerBouquet.png"

try:
    draw_image(image_path, 0, 0)
    screen.mainloop()  # Keeps the window responsive after drawing completes
except KeyboardInterrupt:
    pass
finally:
    try:
        turtle.bye()
    except Exception:
        pass
