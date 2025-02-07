INPUT_SIZE = 10
HIDDEN_SIZE = 16
OUTPUT_SIZE = 4
POPULATION_SIZE = 100
MUTATION_RATE = 0.15
GENERATIONS = 10000


import pygame
import numpy as np
import random



class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.activations = []

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
            self.biases.append(np.random.randn(layer_sizes[i + 1]) * 0.1)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, x):
        self.activations = [x]
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.softmax(z)
            self.activations.append(a)
        return self.activations[-1]

    def save(self, filename):
        save_dict = {'layer_sizes': np.array(self.layer_sizes)}
        for i, w in enumerate(self.weights):
            save_dict[f'weights_{i}'] = w
        for i, b in enumerate(self.biases):
            save_dict[f'biases_{i}'] = b
        np.savez(filename, **save_dict)

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        layer_sizes = data['layer_sizes'].tolist()
        net = cls(layer_sizes)
        for i in range(len(net.weights)):
            net.weights[i] = data[f'weights_{i}']
            net.biases[i] = data[f'biases_{i}']
        return net


class SnakeGame:
    def __init__(self, width=20, height=20, grid_size=20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (0, 0)
        self.food = self.spawn_food()
        self.score = 0
        self.dead = False
        self.steps_since_last_food = 0  

    def spawn_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def get_state(self):
        head = self.snake[0]
        state = []

        
        state.append((self.food[0] - head[0]) / self.width)
        state.append((self.food[1] - head[1]) / self.height)

        
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in dirs:
            next_pos = (head[0] + dx, head[1] + dy)
            if (next_pos in self.snake or
                    next_pos[0] < 0 or next_pos[0] >= self.width or
                    next_pos[1] < 0 or next_pos[1] >= self.height):
                state.append(1)
            else:
                state.append(0)

        
        direction = self.direction
        direction_one_hot = [
            1 if direction == (0, -1) else 0,  
            1 if direction == (0, 1) else 0,  
            1 if direction == (-1, 0) else 0,  
            1 if direction == (1, 0) else 0  
        ]
        state += direction_one_hot

        return np.array(state)

    def step(self, action):
        if self.dead:
            return

        
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_dir = dirs[action]

        
        if (new_dir[0] != -self.direction[0] or new_dir[1] != -self.direction[1]):
            self.direction = new_dir

        
        new_head = (self.snake[0][0] + self.direction[0],
                    self.snake[0][1] + self.direction[1])

        
        if (new_head in self.snake or
                new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height):
            self.dead = True
            return

        self.snake.insert(0, new_head)

        
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            self.steps_since_last_food = 0  
        else:
            self.snake.pop()
            self.steps_since_last_food += 1  

        
        if self.steps_since_last_food > 100:  
            self.dead = True


class Visualizer:
    def __init__(self, game, network, headless=False):
        self.game = game
        self.network = network
        self.headless = headless

        if not self.headless:
            pygame.init()
            self.game_width = game.width * game.grid_size
            self.game_height = game.height * game.grid_size
            self.network_width = 400
            self.window = pygame.display.set_mode((self.game_width + self.network_width,
                                                   max(game.height * game.grid_size, 400)))
            self.clock = pygame.time.Clock()

    def draw_network(self):
        layer_spacing = 100
        neuron_radius = 15
        x_start = self.game_width + 50

        
        for i in range(len(self.network.weights)):
            for j in range(self.network.layer_sizes[i]):
                for k in range(self.network.layer_sizes[i + 1]):
                    weight = self.network.weights[i][j][k]
                    color = (255, 0, 0) if weight < 0 else (0, 0, 255)
                    alpha = min(255, int(abs(weight) * 255))
                    y1 = 50 + j * (self.window.get_height() - 100) / self.network.layer_sizes[i]
                    y2 = 50 + k * (self.window.get_height() - 100) / self.network.layer_sizes[i + 1]
                    line_surface = pygame.Surface((layer_spacing, 1), pygame.SRCALPHA)
                    line_surface.fill((*color, alpha))
                    self.window.blit(line_surface, (x_start + i * layer_spacing, y1))
                    pygame.draw.line(self.window, (*color, alpha),
                                     (x_start + i * layer_spacing, y1),
                                     (x_start + (i + 1) * layer_spacing, y2))

        for i, layer in enumerate(self.network.activations):
            layer_max = np.max(layer) if np.max(layer) != 0 else 1.0
            for j in range(len(layer)):
                
                activation = np.clip(layer[j] / layer_max, 0.0, 1.0)

                
                green_intensity = int(255 * activation)
                color = (0, green_intensity, 0)  

                
                y = 50 + j * (self.window.get_height() - 100) / len(layer)
                x = x_start + i * layer_spacing

                pygame.draw.circle(self.window, color, (int(x), int(y)), neuron_radius)
                pygame.draw.circle(self.window, (255, 255, 255), (int(x), int(y)), neuron_radius, 1)

    def draw_game(self):
        self.window.fill((0, 0, 0))

        
        for i, (x, y) in enumerate(self.game.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(self.window, color,
                             (x * self.game.grid_size, y * self.game.grid_size,
                              self.game.grid_size - 1, self.game.grid_size - 1))

        
        pygame.draw.rect(self.window, (255, 0, 0),
                         (self.game.food[0] * self.game.grid_size,
                          self.game.food[1] * self.game.grid_size,
                          self.game.grid_size - 1, self.game.grid_size - 1))

    def update(self):
        if self.headless:
            return  

        self.draw_game()
        self.draw_network()
        pygame.display.flip()
        self.clock.tick(10)  


def mutate(net):
    for i in range(len(net.weights)):
        mask = np.random.rand(*net.weights[i].shape) < MUTATION_RATE
        net.weights[i] += np.random.randn(*net.weights[i].shape) * 0.1 * mask

        mask = np.random.rand(*net.biases[i].shape) < MUTATION_RATE
        net.biases[i] += np.random.randn(*net.biases[i].shape) * 0.1 * mask
    return net


def train_population(population, game, generations=1000, headless=True):
    visualizer = Visualizer(game, None, headless=headless)
    best_model = None
    best_score = -1

    for generation in range(generations):
        fitness_scores = []
        for net in population:
            game.reset()
            steps = 0
            while not game.dead:
                state = game.get_state()
                output = net.forward(state)
                action = np.argmax(output)
                game.step(action)

                if not headless:
                    visualizer.network = net
                    visualizer.update()
                steps += 1

            fitness = game.score * 100 - steps
            fitness_scores.append(fitness)

            if fitness > best_score:
                best_score = fitness
                best_model = net

        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_nets = [population[i] for i in sorted_indices[:5]]

        new_population = [best_model]

        for _ in range(POPULATION_SIZE - 1):
            parent = random.choice(top_nets)
            child = NeuralNetwork(parent.layer_sizes)
            child.weights = [w.copy() for w in parent.weights]
            child.biases = [b.copy() for b in parent.biases]
            new_population.append(mutate(child))

        population = new_population

        print(f"Generation {generation + 1}, Best Score: {best_score}")

        if generation % 50 == 0:
            best_model.save('snake_model_backup.npz')
            print("Checkpoint: Model saved!")
    best_model.save('snake_model.npz')
    return population


if __name__ == "__main__":
    game = SnakeGame()
    population = [NeuralNetwork([INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE])
                  for _ in range(POPULATION_SIZE)]

    load_model = input("Load existing model? (y/n): ").lower() == 'y'
    best_net = None

    if load_model:
        try:
            best_net = NeuralNetwork.load('snake_model.npz')
            print("Model loaded successfully!")
            population[0] = best_net
        except:
            print("No main model found, trying backup...")
            try:
                best_net = NeuralNetwork.load('snake_model.npz')
                print("Backup model loaded successfully!")
                population[0] = best_net
            except:
                print("No backup model found, training new population")
    train_model = input("Train model? (y/n): ").lower() == 'y'
    if train_model:
        trained_population = train_population(population, game, generations=GENERATIONS, headless=True)
        best_net = max(trained_population, key=lambda net: game.score)

    print("Training complete. Visualizing best network...")
    visualizer = Visualizer(game, best_net, headless=False)
    auto_reset_timer = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                best_net.save('snake_model.npz')
                best_net.save('snake_model.npz')
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    best_net.save('snake_model.npz')
                    best_net.save('snake_model.npz')
                    print("Model saved successfully!")
                elif event.key == pygame.K_l:
                    try:
                        best_net = NeuralNetwork.load('snake_model.npz')
                        visualizer.network = best_net
                        game.reset()
                        print("Model loaded successfully!")
                    except:
                        print("Error loading model!")

        if game.dead:
            auto_reset_timer += 1
            if auto_reset_timer > 10:
                game.reset()
                auto_reset_timer = 0
        else:
            state = game.get_state()
            output = best_net.forward(state)
            action = np.argmax(output)
            game.step(action)

        visualizer.update()
