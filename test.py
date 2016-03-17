from ale_python_interface.ale_python_interface import ALEInterface
import sys
import random
import numpy as np
from nt.visualization import context_manager, facet_grid
from nt.nn import NetOutInspector, LoggerMonitor, ScatterPlotter, RunningAverageMonitor, CollectEpisodeAverageMonitor, NormTransformer, LinePlotter
from nt.nn import VariableGradientInspector

from nt.nn.q_trainer import QNNTrainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.optimizers import SGD
from PIL import Image

class TestNN(Chain):
    def __init__(self, input_size, output_size):
        super(TestNN, self).__init__(
            conv1=L.Convolution2D(1, 32, 8, 4),
            conv2=L.Convolution2D(32, 64, 4, 2),
            conv3=L.Convolution2D(64, 12, 3, 1),
            recurrent=L.LSTM(648, output_size)
        )
        #self.conv1.W.data.fill(0)
        #self.conv2.W.data.fill(0)
        #self.conv3.W.data.fill(0)
        self.output_size = output_size
        self.recurrentState = None, None

    def _propagate(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        #h = F.max_pooling_2d(h, 3, 1, pad=1)
        h = F.relu(self.recurrent(h))
        return h

    def reset_state(self):
        self.recurrent.reset_state()
        self.recurrentState = None, None

    def predict(self, **kwargs):

        epsilon = kwargs.pop('epsilon')
        # epsilon greedy exploration
        if random.uniform(0,1) < epsilon:
            # random output
            return np.array([random.uniform(0,20) for i in range(self.output_size)], dtype=np.float32)

        self.recurrent.c, self.recurrent.h = self.recurrentState
        x = kwargs.pop('x')
        y = self._propagate(x)
        self.recurrentState = self.recurrent.c, self.recurrent.h
        return y.data

    def forward(self, epsilon, **kwargs):
        #reset the recurrent layer before a random minibatch update
        self.recurrent.reset_state()
        x = kwargs.pop('x')
        targets = kwargs.pop('targets')

        y = self._propagate(x)
        loss = F.mean_squared_error(y, targets)
        return loss

def show_screen(resolution, screen):
    img = Image.new( 'RGB', (resolution[0],resolution[1]), "black") # create a new black image
    pixels = img.load() # create the pixel map
    i = 0
    for x in range(img.size[1]):    # for every pixel:
        for y in range(img.size[0]):
            pixels[y,x] = (screen[i],screen[i],screen[i]) # set the colour accordingly
            i += 1
    img.show()

show_screen = True
training_episodes = 1
cv_episodes = 1
epochs = 10000
epsilon = 0.6 # for epsilon greedy
frame_skipping = 4
game_name = 'space_invaders'

ale = ALEInterface()
ale.setBool('display_screen', show_screen)
ale.setBool('color_averaging', True)
ale.setInt("frame_skip",frame_skipping)
ale.setInt("random_seed", random.randint(0,100000))
ale.loadROM('/home/jflick/ale/roms/'+game_name+'.bin')
# Get the list of legal actions
legal_actions = ale.getMinimalActionSet()
resolution = ale.getScreenDims()
dimension = resolution[0] * resolution[1]
reduced_dimension = int(resolution[0]/2 * resolution[1]/2)

def preprocess_screen(screen):
    reshaped = np.reshape(screen, (resolution[1],resolution[0]))

    newscreen = []
    for x in range(0,resolution[1],2):
        row = []
        for y in range(0,resolution[0],2):
            row.append((float(reshaped[x][y])+float(reshaped[x][y+1])+float(reshaped[x+1][y])+float(reshaped[x+1][y+1]))/4)
        newscreen.append(row)
    return np.array([newscreen], dtype=np.uint8)

def get_screen():
    if ale.game_over():
        return None
    screen = np.empty(dimension, dtype=np.uint8)
    ale.getScreen(screen)
    screen = np.reshape(screen, (resolution[1],resolution[0]))
    return preprocess_screen(screen).astype(np.float32)
def get_reward(action):
    return ale.act(action)
'''
def get_screen():
    if random.randint(0,20) == 0:
        return None
    screen = np.array([random.randint(0,100) for i in range(dimension)], dtype=np.uint8)
    ale.getScreen(screen)
    screen = np.reshape(screen, (resolution[1],resolution[0]))
    return preprocess_screen(screen).astype(np.float32)
def get_reward(action):
    return random.uniform(0,20)
'''

nn = TestNN(reduced_dimension, len(legal_actions))

trainer = QNNTrainer(nn,
                     get_screen,
                     get_reward,
                     SGD(),
                     game_name.capitalize() + ' Learner',
                     game_name,
                     nn.forward,
                     nn.forward,
                     nn.predict,
                     action_set=legal_actions,
                     epochs=epochs,
                     use_gpu=False,
                     run_in_thread=True,
                     retain_gradients=True,
                     resume=True,
                     update_interval=50,
                     train_kwargs={'epsilon': epsilon},
                     cv_kwargs={'epsilon': 0.},
                     train_episodes = training_episodes,
                     cv_episodes = cv_episodes,
                     patience = 100)

def prepare_episode():
    ale.reset_game()
    nn.reset_state()

trainer.register_pre_cv_episode_fcn(prepare_episode)
trainer.register_pre_train_episode_fcn(prepare_episode)
trainer.test_run(1)
net_out = trainer.current_net_out
graph = trainer.get_computational_graph()

reward_inspector = NetOutInspector('reward', net_out=net_out)

epoch_reward_mon_tr = CollectEpisodeAverageMonitor(
    'reward mon tr epoch',
    (reward_inspector),
    ('reward'),
    LinePlotter()
)

trainer.add_tr_monitor(epoch_reward_mon_tr)

trainer.start_training()
trainer.print_training_status()
