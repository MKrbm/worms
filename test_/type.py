import pygame
import random

pygame.init()

# Set the width and height of the screen
size = (800, 600)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Typing Game")

# Set the font for the text
font = pygame.font.Font(None, 36)

# Set the colors
white = (255, 255, 255)
black = (0, 0, 0)

# Set the text to be typed
text_list = ["Hello", "World", "Python", "Programming", "Typing", "Game"]
current_text = random.choice(text_list)

# Set the starting position for the text
text_x = 350
text_y = 250

# Set the starting score
score = 0

# Set the clock
clock = pygame.time.Clock()

# Set the game loop
done = False

while not done:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                current_text = current_text[:-1]
            elif event.key == pygame.K_RETURN:
                if current_text == text:
                    score += 1
                    current_text = random.choice(text_list)
            else:
                current_text += pygame.key.name(event.key)

    # Fill the background color
    screen.fill(white)

    # Draw the text
    text = font.render(current_text, True, black)
    screen.blit(text, [text_x, text_y])

    # Draw the score
    score_text = font.render("Score: " + str(score), True, black)
    screen.blit(score_text, [10, 10])

    # Update the screen
    pygame.display.flip()

    # Set the frame rate
    clock.tick(60)

# Quit the game
pygame.quit()