"""
CityMind – Main Entry Point
Run this file to launch the city simulation.

    python main.py

Requirements: pip install pygame
"""
import pygame
import sys
import os

# Ensure we can find our modules
sys.path.insert(0, os.path.dirname(__file__))

from graph import CityGraph
from renderer import CityRenderer
from ui import UIManager


def main():
    pygame.init()
    pygame.display.set_caption("CityMind – Urban Intelligence System")

    # Fullscreen
    info = pygame.display.Info()
    sw, sh = info.current_w, info.current_h
    screen = pygame.display.set_mode((sw, sh), pygame.FULLSCREEN | pygame.HWSURFACE)

    # Shared graph
    graph = CityGraph()

    # Renderer
    renderer = CityRenderer(screen, graph)
    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    renderer.load_sprites(asset_dir)

    # UI (also handles CSP + MST + all interactions)
    ui = UIManager(screen, graph, renderer)

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                if event.key == pygame.K_ESCAPE and not ui.popup and not ui.ctx_menu:
                    pygame.quit()
                    sys.exit()

            ui.handle_event(event)

        # Draw everything
        ui.draw()

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
