import pygame

pygame.init()
screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Hello folks!")
clock = pygame.time.Clock()

game_over = False


while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
    
    screen.fill("blue")

    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()