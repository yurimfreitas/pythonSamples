import pygame

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
sprite1 = pygame.image.load('C://Users//yfreitas//Documents//pythonSamples//football.png')
sprite2 = pygame.image.load('C://Users//yfreitas//Documents//pythonSamples//butterfly.png')
sprite3 = pygame.transform.scale(sprite2, (32,32))
pygame.display.set_caption("Hello folks!")

game_over = False

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
    
    screen.fill("blue")

    screen.blit(sprite1, (640, 360))
    screen.blit(sprite2, (100,100))
    screen.blit(sprite3, (500,500))
    pygame.display.update()

    #pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()