import pygame

pygame.init()
pygame.joystick.init()

count = pygame.joystick.get_count()
print(f"Found {count} joystick(s)")

for i in range(count):
    j = pygame.joystick.Joystick(i)
    j.init()
    print(f"{i}: {j.get_name()}")
