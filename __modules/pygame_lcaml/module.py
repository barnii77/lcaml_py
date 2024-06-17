import pygame
from pyffi import interface
import lcaml_expression

# Initialize Pygame
pygame.init()


# Screen-related functions
@interface(name="set_mode")
def set_mode(size: list, flags: int = 0, depth: int = 0):
    return pygame.display.set_mode(size, flags, depth)


@interface(name="set_caption")
def set_caption(title: str, icontitle: str = None):
    pygame.display.set_caption(title, icontitle)


@interface(name="flip")
def flip(_=None):
    pygame.display.flip()


# Event handling functions
@interface(name="get")
def get(_=None):
    return pygame.event.get()


@interface(name="quit")
def quit(_=None):
    pygame.quit()


# Drawing functions
@interface(name="fill")
def fill(surface, color: list, rect=None):
    surface.fill(color, rect)


@interface(name="draw_rect")
def draw_rect(surface, color: list, rect: list, width: int = 0):
    pygame.draw.rect(surface, color, rect, width)


@interface(name="draw_circle")
def draw_circle(surface, color: list, center: list, radius: int, width: int = 0):
    pygame.draw.circle(surface, color, center, radius, width)


@interface(name="draw_polygon")
def draw_polygon(surface, color: list, points: list, width: int = 0):
    pygame.draw.polygon(surface, color, points, width)


# Time-related functions
@interface(name="wait")
def wait(milliseconds: int):
    pygame.time.wait(milliseconds)


@interface(name="get_ticks")
def get_ticks(_=None):
    return pygame.time.get_ticks()


@interface(name="Clock")
def clock(_=None):
    return pygame.time.Clock()


@interface(name="tick")
def tick(clock, fps: int):
    clock.tick(fps)


# Key-related functions
@interface(name="get_pressed")
def get_pressed(_=None):
    return pygame.key.get_pressed()


@interface(name="key_code")
def key_code(key: str):
    return eval(f"pygame.K_{key.lower() if key.isalpha() else key}", {"pygame": pygame})


# Mouse-related functions
@interface(name="get_pos")
def get_pos(_=None):
    return pygame.mouse.get_pos()


@interface(name="set_pos")
def set_pos(pos: list):
    pygame.mouse.set_pos(pos)


# Constants
KEYDOWN = pygame.KEYDOWN
KEYUP = pygame.KEYUP
QUIT = pygame.QUIT

# Create the LML_EXPORTS dictionary to export functions
LML_EXPORTS = {
    "pygame": {
        "quit": quit,
        "display": {
            "set_mode": set_mode,
            "set_caption": set_caption,
            "flip": flip,
        },
        "event": {
            "get": get,
        },
        "draw": {
            "fill": fill,
            "rect": draw_rect,
            "circle": draw_circle,
            "polygon": draw_polygon,
        },
        "time": {
            "wait": wait,
            "get_ticks": get_ticks,
            "Clock": clock,
            "tick": tick,
        },
        "key": {
            "get_pressed": get_pressed,
            "key_code": key_code,
        },
        "mouse": {
            "get_pos": get_pos,
            "set_pos": set_pos,
        },
        "constants": {
            "KEYDOWN": KEYDOWN,
            "KEYUP": KEYUP,
            "QUIT": QUIT,
        }
    }
}
