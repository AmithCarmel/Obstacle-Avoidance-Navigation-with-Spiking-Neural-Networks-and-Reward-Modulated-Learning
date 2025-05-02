# pure_stdp.py
from brian2 import *
import numpy as np
import pygame, random
import matplotlib.pyplot as plt


def load_obstacles(filename="OBSTACLES_3.txt"):
    with open(filename, 'r') as f:
        return [tuple(map(int, line.strip()[1:-1].split(','))) for line in f]

obstacles = load_obstacles()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
GRID_SIZE = 10
BLOCK_SIZE = SCREEN_WIDTH // GRID_SIZE
WHITE, RED, BLUE, BLACK = (255,255,255), (255,0,0), (0,0,255), (0,0,0)
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pure STDP")

moves = [(-1,0), (1,0), (0,-1), (0,1)]

def draw(state):
    screen.fill(WHITE)
    for x in range(0, SCREEN_WIDTH, BLOCK_SIZE):
        pygame.draw.line(screen, BLACK, (x,0), (x,SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
        pygame.draw.line(screen, BLACK, (0,y), (SCREEN_WIDTH,y))
    for ox, oy in obstacles:
        pygame.draw.rect(screen, RED, (oy*BLOCK_SIZE, ox*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(screen, BLUE, (state[1]*BLOCK_SIZE+5, state[0]*BLOCK_SIZE+5, BLOCK_SIZE-10, BLOCK_SIZE-10))
    pygame.display.flip()


def get_obstacle_flags(pos):
    x,y = pos
    return [not (0 <= x+dx < GRID_SIZE and 0 <= y+dy < GRID_SIZE) or (x+dx,y+dy) in obstacles for dx,dy in moves]

def encode_rel_spikes(flags):
    return [(i, (1 if f else 5)*ms + rand()*0.5*ms) for i,f in enumerate(flags)]

def random_start():
    while True:
        p = (random.randint(0,GRID_SIZE-1), random.randint(0,GRID_SIZE-1))
        if p not in obstacles:
            return p

if __name__ == '__main__':
    start_scope()
    prefs.codegen.target = 'numpy'

    n_inputs, n_outputs = 4, 4
    tau_pre, tau_post = 20*ms, 20*ms
    A_plus, A_minus = 0.005, -0.006

    inp = SpikeGeneratorGroup(n_inputs, [], []*ms)
    out = NeuronGroup(n_outputs, 'dv/dt = -v/(10*ms):1', threshold='v>1', reset='v=0', method='exact')
    stdp = Synapses(inp, out,
        '''
        w : 1
        dpre_trace/dt = -pre_trace/tau_pre : 1 (clock-driven)
        dpost_trace/dt = -post_trace/tau_post : 1 (clock-driven)
        ''',
        on_pre='''
            v_post += w
            pre_trace += 1
            w = clip(w + A_plus * post_trace, 0, 1)
        ''',
        on_post='''
            post_trace += 1
            w = clip(w + A_minus * pre_trace, 0, 1)
        ''')
    stdp.connect()
    stdp.w = '0.3 + 0.4*rand()'

    spike_mon = SpikeMonitor(out)
    episodes = 100
    accuracy_history = []
    loss_history = []

    for ep in range(episodes):
        state = random_start()
        correct = total = collisions = 0
        for _ in range(30):
            draw(state)
            flags = get_obstacle_flags(state)
            rel = encode_rel_spikes(flags)
            t0 = defaultclock.t
            inds = [i for i, t in rel]
            times = [t0 + t for _, t in rel]
            inp.set_spikes(inds, times)
            run(10*ms)
            trains = spike_mon.spike_trains()
            spikes = [len(trains[i][trains[i] > t0]) for i in range(n_outputs)]
            action = np.argmax(spikes) if sum(spikes) > 0 else random.randrange(n_outputs)
            dx, dy = moves[action]
            nxt = (state[0] + dx, state[1] + dy)
            total += 1
            if 0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE and nxt not in obstacles:
                correct += 1
                state = nxt
            else:
                collisions += 1
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); exit()
        acc = 100 * correct / total if total else 0
        loss = collisions / total if total else 1
        accuracy_history.append(acc)
        loss_history.append(loss)
        print(f"[Pure STDP] Ep {ep+1}: Acc={acc:.1f}%, Loss={loss:.2f}")

    pygame.quit()

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(accuracy_history); plt.title('Accuracy'); plt.xlabel('Episode'); plt.ylabel('%'); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(loss_history); plt.title('Loss'); plt.xlabel('Episode'); plt.ylabel('Collision Rate'); plt.grid(True)
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,2))
    for n,times in spike_mon.spike_trains().items():
        plt.vlines(times/ms, n+0.6, n+1.4)
    plt.xlabel('Time (ms)'); plt.ylabel('Neuron'); plt.title('Raster'); plt.show()