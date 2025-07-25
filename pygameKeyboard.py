import pygame
from pygame.locals import *

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
pygame.display.set_caption("Hello folks!")
sprite1 = pygame.image.load('C://Users//yfreitas//Documents//pythonSamples//football.png')
sprite2 = pygame.image.load('C://Users//yfreitas//Documents//pythonSamples//butterfly.png')
sprite3 = pygame.transform.scale(sprite2, (32,32))

button_width = 100
button_height = 50
button_rect = [screen.get_width()/2-button_width/2, screen.get_height()/2-button_height/2, button_width, button_height]
button_font = pygame.font.SysFont('Arial', 20)
button_text = button_font.render('Quit', True, 'black')


# create a font object.
# 1st parameter is the font file
# which is present in pygame.
# 2nd parameter is size of the font
font = pygame.font.Font('freesansbold.ttf', 32)
# create a text surface object,
# on which text is drawn on it.
text = font.render('Yuri', True, 'green', 'blue')


game_over = False

x, y = (0,0)

while not game_over:
    rt = clock.tick(100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        elif event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            x-= sprite3.get_width()/2
            y-= sprite3.get_height()/2
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if button_rect[0] <= x and x <= button_rect[0] + button_rect[2] and button_rect[1] <= y and button_rect[3]:
                game_over = True
        
    pressed = pygame.key.get_pressed()
    if pressed[K_UP]:
        y-= 0.5 * rt
    elif pressed[K_DOWN]:
        y+= 0.5 * rt
    elif pressed[K_RIGHT]:
        x+= 0.5 * rt
    elif pressed[K_LEFT]:
        x-= 0.5 * rt
    elif pressed[K_SPACE]:
        x, y = (0,0)

    if x > (screen.get_width() - sprite3.get_width()):
        x = screen.get_width() - sprite3.get_width()
    elif y > (screen.get_height() - sprite3.get_height()):
        y = screen.get_height() - sprite3.get_height()
    elif x < 0:
        x = 0
    elif y < 0:
        y = 0

    screen.fill("blue")

    if button_rect[0] <= x and x <= button_rect[0] + button_rect[2] and button_rect[1] <= y and button_rect[3]:
       pygame.draw.rect(screen, 'red', button_rect)
    else:
       pygame.draw.rect(screen, 'gray', button_rect)
    screen.blit(button_text, (button_rect[0] + (button_width - button_text.get_width())/2, button_rect[1] + (button_height/2 - button_text.get_height()/2 )))
    screen.blit(text, (400,400))
    screen.blit(sprite1, (600, 300))
    screen.blit(sprite2, (100,100))
    screen.blit(sprite3, (x,y))

    pygame.display.update()

    #pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()