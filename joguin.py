import cv2
import mediapipe as mp
import pygame
import math
import numpy as np
import random
import time
import psutil

pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pacmano")
clock = pygame.time.Clock()

BLACK = (10, 10, 30)
YELLOW = (255, 230, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
PELLET_COLORS = [(255, 255, 255), (255, 100, 100), (100, 255, 100), (255, 200, 50)]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

class Pacman:
    def __init__(self, x, y, radius=20, speed=5):
        self.x, self.y, self.radius, self.speed = x, y, radius, speed
        self.angle = 0
        self.mouth_opening = 0
        self.mouth_dir = 1
        self.eating_timer = 0
    def move_towards(self, tx, ty):
        dx, dy = tx - self.x, ty - self.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist > self.speed:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        self.x = max(self.radius + 20, min(SCREEN_WIDTH - self.radius - 20, self.x))
        self.y = max(self.radius + 20, min(SCREEN_HEIGHT - self.radius - 20, self.y))
        if dist > 0:
            self.angle = math.degrees(math.atan2(-dy, dx))
    def animate(self):
        if self.eating_timer > 0:
            self.eating_timer -= 1
            self.radius = 22
        else:
            self.radius = 20
        self.mouth_opening += self.mouth_dir * 5
        if self.mouth_opening > 45 or self.mouth_opening < 5:
            self.mouth_dir *= -1
    def draw(self, surface):
        start_angle = math.radians(self.mouth_opening)
        end_angle = math.radians(360 - self.mouth_opening)
        pygame.draw.circle(surface, YELLOW, (int(self.x), int(self.y)), self.radius)
        pygame.draw.polygon(surface, BLACK, [
            (self.x, self.y),
            (self.x + math.cos(math.radians(self.angle + self.mouth_opening)) * self.radius,
             self.y - math.sin(math.radians(self.angle + self.mouth_opening)) * self.radius),
            (self.x + math.cos(math.radians(self.angle - self.mouth_opening)) * self.radius,
             self.y - math.sin(math.radians(self.angle - self.mouth_opening)) * self.radius)
        ])

class Pellet:
    def __init__(self, x, y, radius=4, color=None):
        self.x, self.y, self.radius = x, y, radius
        self.color = color if color else random.choice(PELLET_COLORS)
    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius)

def letra_coords(letra, x, y, escala=10):
    img = np.zeros((20, 20), np.uint8)
    cv2.putText(img, letra, (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    pontos = []
    for i in range(20):
        for j in range(20):
            if img[i,j] > 0:
                pontos.append((x + j*escala//2, y + i*escala//2))
    return pontos

player = Pacman(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
pellets = [Pellet(i, j) for i in range(60, SCREEN_WIDTH - 60, 80) for j in range(60, SCREEN_HEIGHT - 60, 80)]
x_offset = 150
for letra in "LASETE":
    for p in letra_coords(letra, x_offset, 250, 12):
        pellets.append(Pellet(p[0], p[1], 4, (255, 200, 50)))
    x_offset += 90

walls = [pygame.Rect(i, j, 40, 10) for i in range(80, SCREEN_WIDTH - 80, 160) for j in range(80, SCREEN_HEIGHT - 80, 160)]

score = 0
font = pygame.font.Font(None, 36)
fps_font = pygame.font.Font(None, 24)
target_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
start_time = time.time()
frames_processed = 0
latencias = []

running = True
while running and cap.isOpened():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    success, image = cap.read()
    if not success:
        continue
    t0 = time.time()
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    cam_h, cam_w, _ = image.shape
    processing_time = (time.time() - t0) * 1000
    latencias.append(processing_time)
    frames_processed += 1
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tx, ty = int(tip.x * SCREEN_WIDTH), int(tip.y * SCREEN_HEIGHT)
            target_pos = (tx, ty)
    old_x, old_y = player.x, player.y
    player.move_towards(*target_pos)
    for wall in walls:
        if wall.colliderect(pygame.Rect(player.x - player.radius, player.y - player.radius, player.radius * 2, player.radius * 2)):
            player.x, player.y = old_x, old_y
            break
    player.animate()
    for pellet in pellets[:]:
        if math.dist((player.x, player.y), (pellet.x, pellet.y)) < player.radius + pellet.radius:
            pellets.remove(pellet)
            score += 10
            player.eating_timer = 5
    screen.fill(BLACK)
    for wall in walls:
        pygame.draw.rect(screen, BLUE, wall)
    for pellet in pellets:
        pellet.draw(screen)
    player.draw(screen)
    fps = int(clock.get_fps())
    uptime = time.time() - start_time
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    avg_lat = np.mean(latencias[-30:]) if latencias else 0
    stats_text = [
        f"FPS: {fps}",
        f"Score: {score}",
        f"Pellets: {len(pellets)}",
        f"CPU: {cpu:.1f}%",
        f"Mem: {mem:.1f}%",
        f"Latency(avg): {avg_lat:.1f} ms",
        f"Frames: {frames_processed}",
        f"Uptime: {uptime:.1f}s"
    ]
    for i, text in enumerate(stats_text):
        t = fps_font.render(text, True, WHITE)
        screen.blit(t, (10, 10 + i * 20))
    pygame.display.flip()
    cv2.imshow('Webcam', image)
    clock.tick(30)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
