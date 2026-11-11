import pygame
import math
import numpy as np
import os

# Initialize Pygame & Fonts
pygame.init()
pygame.font.init()

# Window Setup (1000x700 Fullscreen Track)
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Racing Engine - Evolution Simulator")

# Color Palette
GRASS_GREEN = (34, 139, 34)
ROAD_GREY = (60, 60, 60)
KERB_RED = (200, 30, 30)
KERB_WHITE = (240, 240, 240)
CAR_RED = (220, 20, 60)
CAR_LEAD = (255, 215, 0)
CAR_HUMAN = (0, 225, 255)     # Cyan for Human Player
RAY_YELLOW = (255, 255, 0)
TEXT_WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
PANEL_BG = (20, 22, 28)
SKID_DARK = (20, 20, 20, 140)


# --- 1. NEURAL NETWORK WITH FORWARD BIAS & CROSSOVER ---
class NeuralNetwork:
    def __init__(self, layers=[6, 12, 2]):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.5
            b = np.random.randn(1, layers[i+1]) * 0.5
            
            # Smart Forward Bias: Default output 0 (Throttle) to positive motion
            if i == len(layers) - 2:
                b[0, 0] = 0.8
                
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs):
        a = np.array(inputs, dtype=float).reshape(1, -1)
        for w, b in zip(self.weights, self.biases):
            a = np.tanh(np.dot(a, w) + b)
        return a.flatten()

    def mutate(self, rate=0.15, scale=0.12):
        child = NeuralNetwork(self.layers)
        for i in range(len(self.weights)):
            w_mask = np.random.rand(*self.weights[i].shape) < rate
            w_noise = np.random.randn(*self.weights[i].shape) * scale
            child.weights[i] = self.weights[i] + (w_mask * w_noise)

            b_mask = np.random.rand(*self.biases[i].shape) < rate
            b_noise = np.random.randn(*self.biases[i].shape) * scale
            child.biases[i] = self.biases[i] + (b_mask * b_noise)
        return child

    def save(self, filename="best_brain.npz"):
        """Saves network weights and biases to disk."""
        data = {}
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            data[f"w_{idx}"] = w
            data[f"b_{idx}"] = b
        np.savez(filename, **data)

    def load(self, filename="best_brain.npz"):
        """Loads network weights and biases from disk."""
        if not os.path.exists(filename):
            return False
        data = np.load(filename)
        self.weights = [data[f"w_{i}"] for i in range(len(self.layers) - 1)]
        self.biases = [data[f"b_{i}"] for i in range(len(self.layers) - 1)]
        return True


def crossover(parent1_brain, parent2_brain):
    """Blends 50% of weights and biases from two successful parent brains."""
    child_brain = NeuralNetwork(parent1_brain.layers)
    for i in range(len(parent1_brain.weights)):
        mask_w = np.random.rand(*parent1_brain.weights[i].shape) > 0.5
        child_brain.weights[i] = np.where(mask_w, parent1_brain.weights[i], parent2_brain.weights[i])
        
        mask_b = np.random.rand(*parent1_brain.biases[i].shape) > 0.5
        child_brain.biases[i] = np.where(mask_b, parent1_brain.biases[i], parent2_brain.biases[i])
    return child_brain


# --- 2. TRACK GEOMETRY & PERPENDICULAR CHECKPOINTS ---
track_points = [
    (200, 570), (500, 570), (780, 610), (900, 480), 
    (830, 280), (650, 200), (520, 380), (370, 380), 
    (280, 180), (140, 180), (90, 380)
]

track_surface = pygame.Surface((WIDTH, HEIGHT))
track_surface.fill(GRASS_GREEN)

pygame.draw.lines(track_surface, KERB_RED, True, track_points, 120)
pygame.draw.lines(track_surface, KERB_WHITE, True, track_points, 110)
pygame.draw.lines(track_surface, ROAD_GREY, True, track_points, 96)

# Persistent Skid Mark Overlay Surface
skid_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

# Perpendicular Checkpoints
gates = [
    ((340, 480), (340, 660)),  # Gate 0 (Start/Finish Line)
    ((500, 480), (500, 660)),
    ((740, 510), (820, 710)),
    ((760, 470), (920, 620)),
    ((760, 420), (970, 340)),
    ((710, 160), (770, 320)),
    ((500, 230), (670, 350)),
    ((440, 280), (440, 480)),
    ((220, 330), (430, 230)),
    ((210, 100), (210, 260)),
    ((10, 250),  (220, 310)),
]

def line_intersects_rect(line_p1, line_p2, rect):
    clipline = rect.clipline(line_p1, line_p2)
    return len(clipline) > 0


# --- 3. VEHICLE BASE CLASS (HUMAN & AI) ---
class BaseCar:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.max_speed = 6.0
        self.acceleration = 0.2
        self.friction = 0.96
        self.turn_speed = 3.5
        self.alive = True
        self.on_grass = False
        
        # Metrics
        self.fitness = 0
        self.gates_passed = 0
        self.next_gate_idx = 1
        self.time_alive = 0
        self.time_since_gate = 0
        
        self.current_lap_ticks = 0
        self.last_lap_time = None

        self.trail_history = []

    def get_rect(self):
        return pygame.Rect(int(self.x) - 13, int(self.y) - 7, 26, 14)

    def check_wall_collision(self, surface):
        px, py = int(self.x), int(self.y)
        if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
            self.alive = False
            return
            
        color = surface.get_at((px, py))[:3]
        if color != ROAD_GREY:
            self.alive = False

    def update_physics(self, throttle, steer, surface, human_alive=True):
        if not self.alive:
            return

        self.time_alive += 1
        self.current_lap_ticks += 1

        # Checkpoint timeout active when no human is driving/alive
        if human_alive:
            self.time_since_gate = 0
        else:
            self.time_since_gate += 1
            if self.time_since_gate > 180:  # 3 Seconds @ 60 FPS
                self.alive = False
                return

        self.check_wall_collision(surface)
        if not self.alive:
            return

        effective_max = 3.0 if self.on_grass else self.max_speed
        effective_friction = 0.85 if self.on_grass else self.friction

        self.speed += throttle * self.acceleration
        self.speed = max(-effective_max / 2, min(self.speed, effective_max))
        self.speed *= effective_friction

        if abs(self.speed) > 0.1:
            direction = 1 if self.speed > 0 else -1
            self.angle += steer * self.turn_speed * direction

        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

        self.trail_history.append((int(self.x), int(self.y)))
        if len(self.trail_history) > 15:
            self.trail_history.pop(0)

        if abs(steer) > 0.4 and self.speed > 2.5:
            rear_x = self.x - math.cos(rad) * 10
            rear_y = self.y - math.sin(rad) * 10
            pygame.draw.circle(skid_surface, SKID_DARK, (int(rear_x), int(rear_y)), 3)

        car_rect = self.get_rect()
        target_gate = gates[self.next_gate_idx]
        if line_intersects_rect(target_gate[0], target_gate[1], car_rect):
            self.gates_passed += 1
            self.time_since_gate = 0
            
            if self.next_gate_idx == 0:
                lap_seconds = self.current_lap_ticks / 60.0
                self.last_lap_time = lap_seconds
                self.current_lap_ticks = 0

            self.next_gate_idx = (self.next_gate_idx + 1) % len(gates)

        self.fitness = (self.gates_passed * 1000) + self.time_alive


class HumanCar(BaseCar):
    def check_wall_collision(self, surface):
        px, py = int(self.x), int(self.y)
        if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
            self.alive = False
            return
            
        color = surface.get_at((px, py))[:3]
        if color != ROAD_GREY:
            self.on_grass = True
        else:
            self.on_grass = False

    def update(self, keys, surface, can_move=True):
        if not self.alive or not can_move:
            return

        throttle = 0
        steer = 0
        
        if keys[pygame.K_w]:
            throttle = 1.0
        elif keys[pygame.K_s]:
            throttle = -1.0
            
        if keys[pygame.K_a]:
            steer = -1.0
        elif keys[pygame.K_d]:
            steer = 1.0

        self.update_physics(throttle, steer, surface, human_alive=True)

    def draw(self, surface):
        if not self.alive:
            return
            
        if len(self.trail_history) > 1:
            pygame.draw.lines(surface, (0, 225, 255), False, self.trail_history, 2)

        car_rect = pygame.Surface((26, 14), pygame.SRCALPHA)
        car_rect.fill(CAR_HUMAN)
        rotated_car = pygame.transform.rotate(car_rect, -self.angle)
        new_rect = rotated_car.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated_car, new_rect.topleft)


class AICar(BaseCar):
    def __init__(self, x, y, brain=None):
        super().__init__(x, y)
        self.brain = brain if brain else NeuralNetwork()
        self.ray_angles = [-60, -30, 0, 30, 60]
        self.ray_distances = [0] * 5
        self.max_ray_dist = 220

    def cast_rays(self, surface):
        hit_points = []
        for i, rel_angle in enumerate(self.ray_angles):
            ray_rad = math.radians(self.angle + rel_angle)
            dist = 0
            
            while dist < self.max_ray_dist:
                dist += 2
                px = int(self.x + math.cos(ray_rad) * dist)
                py = int(self.y + math.sin(ray_rad) * dist)

                if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
                    break

                color = surface.get_at((px, py))[:3]
                if color != ROAD_GREY:
                    break

            self.ray_distances[i] = dist
            hit_points.append((int(self.x + math.cos(ray_rad) * dist), 
                               int(self.y + math.sin(ray_rad) * dist)))
            
        return hit_points

    def update(self, surface, can_move=True, human_alive=True):
        if not self.alive or not can_move:
            return

        self.cast_rays(surface)
        norm_rays = [d / self.max_ray_dist for d in self.ray_distances]
        norm_speed = self.speed / self.max_speed
        nn_inputs = norm_rays + [norm_speed]

        throttle, steer = self.brain.forward(nn_inputs)
        self.update_physics(throttle, steer, surface, human_alive=human_alive)

    def draw(self, surface, is_lead=False):
        if not self.alive:
            return

        trail_color = GOLD if is_lead else (200, 50, 50)
        if len(self.trail_history) > 1:
            pygame.draw.lines(surface, trail_color, False, self.trail_history, 1)

        car_color = CAR_LEAD if is_lead else CAR_RED
        car_rect = pygame.Surface((26, 14), pygame.SRCALPHA)
        car_rect.fill(car_color)
        rotated_car = pygame.transform.rotate(car_rect, -self.angle)
        new_rect = rotated_car.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated_car, new_rect.topleft)


# --- 4. GENETIC ALGORITHM MANAGER ---
POP_SIZE = 20
SPAWN_X, SPAWN_Y = 340, 570

def create_next_generation(old_cars, gen_count=1):
    sorted_cars = sorted(old_cars, key=lambda c: c.fitness, reverse=True)
    parents = sorted_cars[:5]
    new_cars = []
    
    # Elitism
    new_cars.append(AICar(SPAWN_X, SPAWN_Y, brain=parents[0].brain))
    
    rate = 0.25 if gen_count < 5 else 0.08
    scale = 0.25 if gen_count < 5 else 0.08
    
    while len(new_cars) < POP_SIZE:
        p1, p2 = np.random.choice(parents, size=2, replace=False)
        child_brain = crossover(p1.brain, p2.brain)
        child_brain = child_brain.mutate(rate=rate, scale=scale)
        new_cars.append(AICar(SPAWN_X, SPAWN_Y, brain=child_brain))
        
    return new_cars


# --- 5. MAIN MENU RENDERER ---
def draw_main_menu(surface, font_large, font_medium, font_small):
    surface.fill(PANEL_BG)

    title = font_large.render("AI RACING SIMULATOR", True, GOLD)
    subtitle = font_medium.render("Select Simulation Mode", True, TEXT_WHITE)

    surface.blit(title, (WIDTH // 2 - title.get_width() // 2, 120))
    surface.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 200))

    # Mode 1 Card
    card1_rect = pygame.Rect(180, 280, 280, 220)
    pygame.draw.rect(surface, (32, 36, 45), card1_rect, border_radius=12)
    pygame.draw.rect(surface, GOLD, card1_rect, 2, border_radius=12)

    c1_title = font_medium.render("[ 1 ] Solo AI Training", True, GOLD)
    c1_desc1 = font_small.render("• Pure Autonomous Evolution", True, TEXT_WHITE)
    c1_desc2 = font_small.render("• Hands-free fast training", True, TEXT_WHITE)
    c1_desc3 = font_small.render("• Active 3-second culling", True, TEXT_WHITE)

    surface.blit(c1_title, (card1_rect.x + 20, card1_rect.y + 25))
    surface.blit(c1_desc1, (card1_rect.x + 20, card1_rect.y + 80))
    surface.blit(c1_desc2, (card1_rect.x + 20, card1_rect.y + 115))
    surface.blit(c1_desc3, (card1_rect.x + 20, card1_rect.y + 150))

    # Mode 2 Card
    card2_rect = pygame.Rect(540, 280, 280, 220)
    pygame.draw.rect(surface, (32, 36, 45), card2_rect, border_radius=12)
    pygame.draw.rect(surface, CAR_HUMAN, card2_rect, 2, border_radius=12)

    c2_title = font_medium.render("[ 2 ] Human vs AI", True, CAR_HUMAN)
    c2_desc1 = font_small.render("• WASD Manual Control", True, TEXT_WHITE)
    c2_desc2 = font_small.render("• Dynamic timeout rules", True, TEXT_WHITE)
    c2_desc3 = font_small.render("• Off-road slowdowns", True, TEXT_WHITE)

    surface.blit(c2_title, (card2_rect.x + 25, card2_rect.y + 25))
    surface.blit(c2_desc1, (card2_rect.x + 25, card2_rect.y + 80))
    surface.blit(c2_desc2, (card2_rect.x + 25, card2_rect.y + 115))
    surface.blit(c2_desc3, (card2_rect.x + 25, card2_rect.y + 150))

    prompt = font_small.render("Press 1 or 2 on keyboard to select", True, (160, 160, 170))
    surface.blit(prompt, (WIDTH // 2 - prompt.get_width() // 2, 560))


# --- 6. MAIN SIMULATION LOOP ---
clock = pygame.time.Clock()

game_state = 'MENU'       # States: 'MENU', 'RACE'
game_mode = 'SOLO_AI'     # Modes:  'SOLO_AI', 'HUMAN_VS_AI'

ai_cars = [AICar(SPAWN_X, SPAWN_Y) for _ in range(POP_SIZE)]
human_car = None

global_human_best = None
global_ai_best = None

generation = 1
fast_forward = False

race_state = 'COUNTDOWN'
countdown_ticks = 180
status_msg = ""

font_large = pygame.font.Font(None, 52)
font_medium = pygame.font.Font(None, 32)
font_hud = pygame.font.Font(None, 24)
font_small = pygame.font.Font(None, 20)

running = True
while running:
    fps = 300 if fast_forward else 60
    clock.tick(fps)

    # ==================== A. MAIN MENU STATE ====================
    if game_state == 'MENU':
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_1, pygame.K_KP1):
                    game_mode = 'SOLO_AI'
                    game_state = 'RACE'
                    human_car = None
                    ai_cars = [AICar(SPAWN_X, SPAWN_Y) for _ in range(POP_SIZE)]
                    generation = 1
                    race_state = 'RACING'  # Immediate start for solo mode
                    skid_surface.fill((0, 0, 0, 0))  # Clear skid marks on mode enter
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    game_mode = 'HUMAN_VS_AI'
                    game_state = 'RACE'
                    human_car = HumanCar(SPAWN_X, SPAWN_Y - 25)
                    ai_cars = [AICar(SPAWN_X, SPAWN_Y) for _ in range(POP_SIZE)]
                    generation = 1
                    race_state = 'COUNTDOWN'
                    countdown_ticks = 180
                    skid_surface.fill((0, 0, 0, 0))  # Clear skid marks on mode enter

        draw_main_menu(screen, font_large, font_medium, font_hud)
        pygame.display.flip()
        continue

    # ==================== B. ACTIVE RACE STATE ====================
    # 1. EVENT HANDLING
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game_state = 'MENU'
                skid_surface.fill((0, 0, 0, 0))  # Clear skid marks when leaving mode
            elif event.key == pygame.K_f:
                fast_forward = not fast_forward
            elif event.key == pygame.K_r:  # Force Reset
                generation += 1
                ai_cars = create_next_generation(ai_cars, generation)
                if game_mode == 'HUMAN_VS_AI':
                    human_car = HumanCar(SPAWN_X, SPAWN_Y - 25)
                    race_state = 'COUNTDOWN'
                    countdown_ticks = 180
                else:
                    race_state = 'RACING'
            elif event.key == pygame.K_p:  # Save Top Brain
                lead_car = max(ai_cars, key=lambda c: c.fitness)
                lead_car.brain.save("best_brain.npz")
                status_msg = "SAVED: best_brain.npz!"
            elif event.key == pygame.K_l:  # Load Brain
                test_brain = NeuralNetwork()
                if test_brain.load("best_brain.npz"):
                    ai_cars = [AICar(SPAWN_X, SPAWN_Y, brain=test_brain) for _ in range(POP_SIZE)]
                    if game_mode == 'HUMAN_VS_AI':
                        human_car = HumanCar(SPAWN_X, SPAWN_Y - 25)
                        race_state = 'COUNTDOWN'
                        countdown_ticks = 180
                    else:
                        race_state = 'RACING'
                    status_msg = "LOADED: best_brain.npz!"

    # 2. COUNTDOWN LOGIC (Human Mode)
    can_move = (race_state == 'RACING' or race_state == 'WAITING_FOR_HUMAN')
    
    if race_state == 'COUNTDOWN':
        countdown_ticks -= 1
        if countdown_ticks <= 0:
            race_state = 'RACING'

    # 3. UPDATE CARS
    keys = pygame.key.get_pressed()
    
    if game_mode == 'HUMAN_VS_AI' and human_car:
        human_car.update(keys, track_surface, can_move=can_move)
        human_is_alive = human_car.alive
    else:
        human_is_alive = False  # In solo mode, 3-sec timeout is always active

    if race_state == 'RACING':
        for c in ai_cars:
            if c.alive:
                c.update(track_surface, can_move=True, human_alive=human_is_alive)

    alive_ai = [c for c in ai_cars if c.alive]

    # 4. WIN & ROUND RESOLUTION LOGIC

    # --- MODE 1: SOLO AI TRAINING ---
    if game_mode == 'SOLO_AI':
        ai_winner = next((c for c in ai_cars if c.last_lap_time is not None), None)
        if ai_winner:
            if global_ai_best is None or ai_winner.last_lap_time < global_ai_best:
                global_ai_best = ai_winner.last_lap_time
            status_msg = f"AI Finished Lap ({ai_winner.last_lap_time:.2f}s)!"
            generation += 1
            ai_cars = create_next_generation(ai_cars, generation)

        elif len(alive_ai) == 0:
            generation += 1
            ai_cars = create_next_generation(ai_cars, generation)

    # --- MODE 2: HUMAN VS AI RACE ---
    elif game_mode == 'HUMAN_VS_AI' and human_car:
        # Human wins first
        if human_car.last_lap_time is not None and race_state == 'RACING':
            status_msg = "HUMAN WINS GENERATION!"
            if global_human_best is None or human_car.last_lap_time < global_human_best:
                global_human_best = human_car.last_lap_time

            generation += 1
            ai_cars = create_next_generation(ai_cars, generation)
            human_car = HumanCar(SPAWN_X, SPAWN_Y - 25)
            race_state = 'RACING'

        # AI wins first
        ai_winner = next((c for c in ai_cars if c.last_lap_time is not None), None)
        if ai_winner and race_state == 'RACING':
            status_msg = "AI WON! Complete your lap!"
            if global_ai_best is None or ai_winner.last_lap_time < global_ai_best:
                global_ai_best = ai_winner.last_lap_time

            generation += 1
            ai_cars = create_next_generation(ai_cars, generation)
            race_state = 'WAITING_FOR_HUMAN'

        # Transition once human finishes/crashes
        if race_state == 'WAITING_FOR_HUMAN':
            if human_car.last_lap_time is not None or not human_car.alive:
                if human_car.last_lap_time is not None:
                    if global_human_best is None or human_car.last_lap_time < global_human_best:
                        global_human_best = human_car.last_lap_time

                human_car = HumanCar(SPAWN_X, SPAWN_Y - 25)
                race_state = 'COUNTDOWN'
                countdown_ticks = 180

        # All AI crash while human is also dead
        if len(alive_ai) == 0 and race_state == 'RACING':
            if not human_car.alive:
                generation += 1
                ai_cars = create_next_generation(ai_cars, generation)
                human_car = HumanCar(SPAWN_X, SPAWN_Y - 25)
                race_state = 'COUNTDOWN'
                countdown_ticks = 180

    # 5. RENDER FRAME
    screen.blit(track_surface, (0, 0))
    screen.blit(skid_surface, (0, 0))

    for g in gates:
        pygame.draw.line(screen, (100, 100, 100), g[0], g[1], 1)

    lead_ai = max(ai_cars, key=lambda c: c.fitness) if any(c.alive for c in ai_cars) else None

    for c in ai_cars:
        if c.alive:
            c.draw(screen, is_lead=(c == lead_ai))

    if lead_ai and lead_ai.alive and race_state == 'RACING':
        hit_points = lead_ai.cast_rays(track_surface)
        for px, py in hit_points:
            pygame.draw.line(screen, RAY_YELLOW, (int(lead_ai.x), int(lead_ai.y)), (px, py), 2)

    if game_mode == 'HUMAN_VS_AI' and human_car:
        human_car.draw(screen)

    # 6. HUD OVERLAY
    speed_mode = "300 FPS [FAST]" if fast_forward else "60 FPS [NORMAL]"
    mode_tag = "SOLO AI TRAINING" if game_mode == 'SOLO_AI' else "HUMAN VS AI"
    info_str = f"MODE: {mode_tag} | GEN: {generation} | Active AI: {len(alive_ai)}/{POP_SIZE} | Speed: {speed_mode}"
    hud_text = font_hud.render(info_str, True, TEXT_WHITE)
    screen.blit(hud_text, (20, 15))

    keys_str = "('ESC'=Menu | 'F'=Fast | 'P'=Save | 'L'=Load | 'R'=Reset)"
    keys_text = font_small.render(keys_str, True, (180, 180, 190))
    screen.blit(keys_text, (20, 38))

    ai_best = f"{global_ai_best:.2f}s" if global_ai_best else "--"
    
    if game_mode == 'HUMAN_VS_AI' and human_car:
        h_status = "OFF-ROAD (3.0)" if human_car.on_grass else f"{human_car.current_lap_ticks / 60.0:.1f}s"
        h_time = h_status if human_car.alive else "CRASHED OUT"
        h_best = f"{global_human_best:.2f}s" if global_human_best else "--"
        timing_str = f"HUMAN Time: {h_time} (Best: {h_best}) | AI Lead Best Lap: {ai_best}"
    else:
        timing_str = f"AI Lead Best Lap: {ai_best}"
        
    timing_text = font_hud.render(timing_str, True, GOLD)
    screen.blit(timing_text, (20, 58))

    if status_msg:
        msg_txt = font_hud.render(status_msg, True, (0, 225, 255))
        screen.blit(msg_txt, (20, 80))

    if race_state == 'COUNTDOWN' and game_mode == 'HUMAN_VS_AI':
        secs_left = int(math.ceil(countdown_ticks / 60.0))
        cd_str = str(secs_left) if secs_left > 0 else "GO!"
        cd_text = font_large.render(cd_str, True, GOLD)
        screen.blit(cd_text, (WIDTH // 2 - 30, HEIGHT // 2 - 30))

    pygame.display.flip()

pygame.quit()