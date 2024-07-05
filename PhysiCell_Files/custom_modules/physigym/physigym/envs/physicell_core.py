#####
# title: pysigym/envs/physicell_core.py
#
# language: python3
# library: gymnasium v1.0.0a1
#
# date: 2024-spring
# license: BSD-3-Clause
# author: Alexandre Bertin, Elmar Bucher
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# description:
#     gymnasium environment for physicell embedding
# + https://gymnasium.farama.org/main/
# + https://gymnasium.farama.org/main/introduction/create_custom_env/
# + https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
#####


# library
from embedding import physicell
import gymnasium
from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# function

class CorePhysiCellEnv(gymnasium.Env):
    """
    input:
        gymnasium.Env

    output:
        physigym.CorePhysiCellEnv

    run:
        offspring: physigym.ModelPhysiCellEnv

    description:
        this is the core physigym environment class, built on top of the
        gymnasium.Env class. physigym.CorePhysiCellEnv class as such will be
        the base class for every physigym.ModelPhysiCellEnv.

        there should be no need to edit the physigym.CorePhysiCellEnv class.
        model specifics should be captured in the physigym.ModelPhysiCellEnv class.
    """

    ### begin dummy functions ###

    def get_action_space(self):
        sys.exit('get_action_space function to be implemented in physigym.ModelPhysiCellEnv!')


    def get_observation_space(self):
        sys.exit('get_observation_space function to be implemented in physigym.ModelPhysiCellEnv!')


    def get_observation(self):
        sys.exit('get_observation function to be implemented in physigym.ModelPhysiCellEnv!')


    def get_info(self):
        sys.exit('get_info function to be implemented in physigym.ModelPhysiCellEnv!')


    def get_terminated(self):
        sys.exit('get_terminated function to be implemented in physigym.ModelPhysiCellEnv!')


    def get_reward(self):
        sys.exit('get_terminated function to be implemented in physigym.ModelPhysiCellEnv!')


    def get_img(self):
        sys.exit('get_img function to be implemented in physigym.ModelPhysiCellEnv!')

    ### end dummy functions ###


    # metadata
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': None,
    }

    # functions
    def __init__(self, settingxml='config/PhysiCell_settings.xml', figsize=(8, 6), render_mode=None, render_fps=10, verbose=True):
        """
        input:
            settingxml: string; default is 'config/PhysiCell_settings.xml'
                path and filename to the settings.xml file.
                the file will be loaded with lxml and stored at self.x_root.
                therefor all data from the setting.xml file is later on accessible
                via the self.x_root.xpath('//xpath/string/') xpath construct.
                study this source code class for explicit examples.
                for more information about xpath study the following links:
                + https://en.wikipedia.org/wiki/XPath
                + https://www.w3schools.com/xml/xpath_intro.asp
                + https://lxml.de/xpathxslt.html#xpat

            figsize: tuple of floats; default is (8, 6) which is a 4:3 ratio.
                values are in inches (width, height).

            render_mode: string as specified in the metadata or None; default is None.

            render_fps: float or None; default is 10.
                if render_mode is 'human', for every dt_gym step the image,
                specified in the physigym.ModelPhysiCellEnv.get_img() function,
                will be generated and displayed. this frame per second setting
                specifies the time the computer sleeps after the image is
                displayed.
                for example 10[fps] = 1/10[spf] = 0.1 [spf].

            verbose:
                to set standard output verbosity true or false.
                please note, only little from the standard output is coming
                actually from physigym. most of the output comes straight
                from PhysiCell and this setting has no influence over that output.

        output:
            initialized PhysiCell Gymnasium environment.

        run:
            import gymnasium
            import physigym

            env = gymnasium.make('physigym/ModelPhysiCellEnv')

            env = gymnasium.make(
                'physigym/ModelPhysiCellEnv',
                settingxml = 'config/PhysiCell_settings.xml',
                figsize = (8, 6),
                render_mode = None,
                render_fps = 10,
                verbose = True
            )

        description:
            function to initialize the PhysiCell Gymnasium environment.
        """
        # handle verbose
        self.verbose = verbose
        if self.verbose:
            print(f'physigym: initialize environment ...')

        # initialize class whide variables
        if self.verbose:
            print(f'physigym: declare class instance-wide variables.')
        self.episode = -1
        self.step_episode = None
        self.step_env = 0

        # load physicell settings.xml file
        self.settingxml = settingxml
        self.x_tree = etree.parse(self.settingxml)
        if self.verbose:
            print(f'physigym: reading {self.settingxml}')
        self.x_root = self.x_tree.getroot()

        # handle render mode and figsize
        if self.verbose:
            print(f'physigym: declare render settings.')
        assert render_mode is None or render_mode in self.metadata['render_modes'], f"'{render_mode}' is an unknown render_mode. known are {sorted(self.metadata['render_modes'])}, and None."
        self.figsize = figsize
        self.render_mode = render_mode
        self.metadata.update({'render_fps': render_fps})
        if not (self.render_mode is None):
            self.fig, axs = plt.subplots(figsize=self.figsize)

        # handle spaces
        if self.verbose:
            print(f'physigym: declare action and observer space.')
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        # output
        if self.verbose:
            print(f'physigym: ok!')


    def render(self):
        """
        input:
            self.get_img()

        output:
            a_img: numpy array or None
                if self.render_mode is
                None: the function will return None.
                rgb_array or human: the function will return a numpy array,
                    8bit, shape (4,y,x) with red, green, blue, and alpha channel.
        run:
            import gymnasium
            import physigym

            env = gymnasium.make('physigym/ModelPhysiCellEnv', render_mode= None)
            env = gymnasium.make('physigym/ModelPhysiCellEnv', render_mode='human')
            env = gymnasium.make('physigym/ModelPhysiCellEnv', render_mode='rgb_array')

            o_observation, d_info = env.reset()
            env.render()

        description:
            function to render the image into an 8bit numpy array,
            if render_mode is not None.
        """
        if self.verbose :
            print(f'physigym: render {self.render_mode} frame ...')

        # processing
        a_img = None

        if not (self.render_mode is None): # human or rgb_array
            self.fig.canvas.draw()
            a_img = np.array(self.fig.canvas.buffer_rgba(), dtype=np.uint8)

        # output
        if self.verbose :
            print(f'ok!')
        return a_img


    def reset(self, seed=None, options={}):
        """
        input:
            self.get_observation()
            self.get_info()
            self.get_img()

            seed: integer or None; default is None
                seed = None: generate a random seed. seed with this value python and PhyiCell (via the setting.xml file).
                seed < 0: take seed from setting.xml
                seed >= 0: the seed from this value and seed python and PhysiCell (via the setting.xml file).

            options: dictionary or None
                reserved for possible future use.

        output:
            o_observation: structure
                the exact structure has to be
                specified in the get_observation_space function.

            d_info: dictionary
                what information to be captured has to be
                specified in the get_info function.

        run:
            import gymnasium
            import physigym

            env = gymnasium.make('physigym/ModelPhysiCellEnv')

            o_observation, d_info = env.reset()

        description:
            The reset method will be called to initiate a new episode,
            increment episode counter, reset episode step counter.
            You may assume that the step method will not be called
            before the reset function has been called.
        """
        if self.verbose :
            print(f'physigym: reset episode ...')

        # handle random seeding
        if (seed is None):
            i_seed = seed
            self.x_root.xpath('//random_seed')[0].text = str(-1)
            self.x_tree.write(self.settingxml, pretty_print=True)
        # handle setting.xml based seeding
        elif (seed < 0):
            i_seed = int(self.x_root.xpath('//random_seed')[0].text)
            if (i_seed < 0):
                i_seed = None
        # handle gymnasium based seeding
        else: # seed >= 0
            i_seed = seed
            self.x_root.xpath('//random_seed')[0].text = str(i_seed)
            self.x_tree.write(self.settingxml, pretty_print=True)
        # seed self.np_random number generator
        super().reset(seed=i_seed)
        if self.verbose:
            print(f'physigym: seed random number generator with {i_seed}.')

        # update class whide variables
        if self.verbose:
            print(f'physigym: update class instance-wide variables.')
        self.episode += 1
        self.step_episode = 0
        #self.step_env NOP

        # initialize physcell model
        if self.verbose:
            print(f'physigym: declare PhysiCell model instance.')
        os.makedirs(self.x_root.xpath('//save/folder')[0].text, exist_ok=True)
        physicell.start(self.settingxml)

        # observe domain
        if self.verbose:
            print(f'physigym: domain observation.')
        o_observation = self.get_observation()
        d_info = self.get_info()

        # render domain
        if self.verbose:
            print(f'physigym: render {self.render_mode} frame.')
        if not (self.render_mode is None):
            plt.ion()
            self.get_img()
            if (self.render_mode == 'human'): # human
                if not (self.metadata['render_fps'] is None):
                    plt.pause(1 / self.metadata['render_fps'])
            else: # rgb_array
                self.fig.canvas.setVisible(False)

        # output
        if self.verbose:
            print(f'physigym: ok!')
        return o_observation, d_info


    def get_truncated(self):
        """
        input:
            settingxml max_time
            PhysiCell parameter time

        output:
            b_truncated: bool

        run:
            internal function.

        description:
            function to evaluate if the episode reached the max_time specified.
        """
        # processing
        b_truncated = False
        r_time_max = float(self.x_root.xpath('//overall/max_time')[0].text)
        r_time_current = physicell.get_parameter('time')  # achtung: time has to be declared as parameter of type float in the settings.xml file!
        b_truncated = r_time_current >= r_time_max

        # output
        return b_truncated


    def step(self, action):
        """
        input:
            self.get_observation()
            self.get_terminated()
            self.get_truncated()
            self.get_info()
            self.get_reward()
            self.get_img()

            action: dict
                object compatible with the defined action space struct.
                the dictionary keys have to match the parameter,
                custom variable, or custom vector label. the values are
                either single or numpy arrays of bool, integer, float,
                or string values.

        output:
            o_observation: structure
                structure defined by the user in self.get_observation_space().

            r_reward: float or int or bool
                algorithm defined by the user in self.get_reward().

            b_terminated: bool
                algorithm defined by the user in self.get_terminated().

            b_truncated: bool
                algorithm defined in self.get_truncated().

            info: dict
                algorithm defined by the user in self.get_info().

            self.episode: integer
                episode counter.

            self.step_episode: integer
                within an episode step counter.

            self.step_env: integer
                overall episodes step counter.

        run:
            import gymnasium
            import physigym

            env = gymnasium.make('physigym/ModelPhysiCellEnv')

            o_observation, d_info = env.reset()
            o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(action={})

        description:
            function does a dt_gym simulation step:
            apply action, increment the step counters, observes, retrieve reward,
            and finalizes a physicell episode, if episode is terminated or truncated.
        """
        if self.verbose :
            print(f'physigym: taking a dt_gym time step ...')

        # do action
        if self.verbose:
            print(f'physigym: action.')
        for s_action, o_value in action.items():  # action is always a gymnasium composite space dict
            # gymnasium composite space tuple: nop.
            # gymnasium composite space sequences: nop.
            # gymnasium composite space graph: nop.

            # gymnasium action space discrete (boolean, integer)
            # python/physicell api parametre, variable
            if (type(o_value) in {bool, int}):
                try:
                    # try custom_variable
                    physicell.set_variable(s_action, o_value)
                except KeyError:
                    # try parameter
                    try:
                        physicell.set_parameter(s_action, o_value)
                    # error
                    except KeyError:
                        sys.exit(f"Error @ physigym.envs.physicell_core.CorePhysiCellEnv : unprocessable gymnasium discrete action space value detected! {s_action} {o_value} {type(o_value)}.")

            # gymnasium action space text (string)
            # python/physicell api parameter, variable
            elif (type(o_value) in {str}):
                try:
                    # try custom_variable
                    physicell.set_variable(s_action, o_value)
                except KeyError:
                    # try parameter
                    try:
                        physicell.set_parameter(s_action, o_value)
                    # error
                    except KeyError:
                        sys.exit(f"Error @ physigym.envs.physicell_core.CorePhysiCellEnv : unprocessable gymnasium text action space value detected! {s_action} {o_value} {type(o_value)}.")

            # gymnasium action space box (bool, int, float in a numpy array)
            # gymnasium action space multi binary (boolean in a numpy array)
            # gymnasium action space multi discrete (boolean, integer in a numpy array)
            # python/physicell api parameter, variabler, vector
            elif (type(o_value) in {np.ndarray}):
                if (len(o_value.shape) > 1):
                    o_value = o_value[0]
                try:
                    # try vector
                    physicell.set_vector(s_action, list(o_value))
                except KeyError:
                    # try custom_variable
                    try:
                        physicell.set_variable(s_action, o_value)
                    # try parameter
                    except KeyError:
                        try:
                            physicell.set_parameter(s_action, o_value)
                        # error
                        except KeyError:
                            sys.exit(f"Error @ physigym.envs.physicell_core.CorePhysiCellEnv : unprocessable gymnasium box action space value detected! {s_action} {o_value} {type(o_value)}.")

            # error
            else:
                sys.exit(f"Error @ physigym.envs.physicell_core.CorePhysiCellEnv : unprocessable gymnasium action space value detected! {s_action} {o_value} {type(o_value)}.")

        # do dt_gym time step
        if self.verbose:
            print(f'physigym: PhysiCell model step.')
        physicell.step()

        # update class whide variables
        if self.verbose:
            print(f'physigym: update class instance-wide variables.')
        #self.episode NOP
        self.step_episode += 1
        self.step_env += 1

        # get observation
        if self.verbose:
            print(f'physigym: domain observation.')
        o_observation = self.get_observation()
        b_terminated = self.get_terminated()
        b_truncated = self.get_truncated()
        d_info = self.get_info()

        # get revard
        r_reward = self.get_reward()

        # do rendering
        if self.verbose:
            print(f'physigym: render {self.render_mode} frame.')
        if not (self.render_mode is None): # human or rgb_array
            self.get_img()
            if (self.render_mode == 'human') and not (self.metadata['render_fps'] is None):
                plt.pause(1 / self.metadata['render_fps'])

        # check if episode finish
        if b_terminated or b_truncated:
            if self.verbose:
                print(f'physigym: PhysiCell model episode finish by termination ({b_terminated}) or truncation ({b_truncated}).')
            physicell.stop()

        # output
        if self.verbose:
            print(f'physigym: ok!')
        return o_observation, r_reward, b_terminated, b_truncated, d_info


    def close(self):
        """
        input:

        output:

        run:
            import gymnasium
            import physigym

            env = gymnasium.make('physigym/ModelPhysiCellEnv')

            env.close()

        description:
            function to drop shutdown physigym environment.
        """
        if self.verbose :
            print(f'physigym: environment closure ...')

        # processing
        if self.verbose:
            print(f'physigym: Gymnasium PhysiCell model environment is going down.')
        if not (self.render_mode is None):
            plt.close(self.fig)

        # output
        if self.verbose:
            print(f'physigym: ok!')


    def verbose_true(self):
        """
        input:

        output:

        run:
            import gymnasium
            import physigym

            env = gymnasium.make('physigym/ModelPhysiCellEnv')

            env.unwrapped.verbose_true()

        description:
            to set verbosity true after initialization.

            please note, only little from the standard output is coming
            actually from physigym. most of the output comes straight
            from PhysiCell and this setting has no influence over that output.
        """
        print(f'physigym: set env.verbose = True.')
        self.verbose = True


    def verbose_false(self):
        """
        input:

        output:

        run:
            import gymnasium
            import physigym

            env = gymnasium.make('physigym/ModelPhysiCellEnv')

            env.unwrapped.verbose_true()

        description:
            to set verbosity false after initialization.

            please note, only little from the standard output is coming
            actually from physigym. most of the output comes straight
            from PhysiCell and this setting has no influence over that output.
        """
        print(f'physigym: set env.verbose = False.')
        self.verbose = False

