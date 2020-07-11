'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

import os

from baselines.gail import mlp_policy_sawyer # changed to use other construction
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier

# For acktr
from baselines.old_acktr.acktr_cont import old_acktr_learn
from baselines.old_acktr.policies import GaussianMlpPolicy
from baselines.old_acktr.value_functions import NeuralNetValueFunction

import baselines.common as common
from baselines.common.filters import ZFilter


# Integration with robosuite
import robosuite
from robosuite.wrappers import GymWrapper

# For loading policy
import tensorflow as tf
import time

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='SawyerLift') # Default to SawyerLift
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='./log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=1500) # change from -1 to 3000
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=1)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1) # changed def from 1 to 2
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0) # notice policy is trained with entropy 0, lets change this to 0.5, and train to see results
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3) # defaults is 1e-3
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=30)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=1e9) # changed to 5e6
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def get_task_name(args):
    task_name = "grasp"+ "_acktr_rl."
    #if args.pretrained:
    #    task_name += "with_pretrained."
    if args.traj_limitation != np.inf:
        task_name += "transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id.split("-")[0]
    #task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
    #    ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    #task_name += ".seed_" + str(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = robosuite.make(args.env_id,
            ignore_done=True,
            use_camera_obs=False,
            has_renderer=True,
            control_freq=100,
            gripper_visualization=True,
            reward_shaping=True,
            #box_pos = [0.63522776, -0.3287869, 0.82162434], # shift2
            #box_quat=[0.6775825618903728, 0, 0, 0.679425538604203], # shift2
            box_pos = [0.23522776, 0.2287869, 0.82162434], #shift3
            box_quat=[0.3775825618903728, 0, 0, 0.679425538604203], #shift3
            #box_pos = [0.53522776, 0.3287869, 0.82162434], #shift4
            #box_quat=[0.5775825618903728, 0, 0, 0.679425538604203], #shift4
            ) # Switch from gym to robosuite, also add reward shaping to see reach goal

    env = GymWrapper(env) # wrap in the gym environment

    task = 'train'
    #task = 'evaluate'

    
    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy_sawyer.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    #env = bench.Monitor(env, logger.get_dir() and
    #                    osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)

    # Note: taking away the bench monitor wrapping allows rendering
    
    #env.seed(args.seed) # Sawyer does not have seed 

    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)

    logger.log("log_directories: ",args.log_dir)
    logger.log("environment action space range: ", env.action_space) #logging the action space
    
    #------- Run policy for reaching ---------#
    play_env = robosuite.make(args.env_id,
            ignore_done=True,
            use_camera_obs=False,
            has_renderer=True,
            control_freq=100,
            gripper_visualization=True,
            #box_pos = [0.63522776, -0.3287869, 0.82162434], # shift2
            #box_quat=[0.6775825618903728, 0, 0, 0.679425538604203], # shift2
            box_pos = [0.23522776, 0.2287869, 0.82162434], #shift3
            box_quat=[0.3775825618903728, 0, 0, 0.679425538604203], #shift3
            #box_pos = [0.53522776, 0.3287869, 0.82162434], #shift4
            #box_quat=[0.5775825618903728, 0, 0, 0.679425538604203], #shift4
            )

    play_env = GymWrapper(play_env)
    
    #Weights are loaded from reach model grasp_strange


    
    #play_env.viewer.set_camera(camera_id=2) # Switch views for eval
    
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi_reach = policy_fn("pi", ob_space, ac_space, reuse=False)

    # Hack for loading policies using tensorflow
    init_op = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(max_to_keep=5)
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Load Checkpoint
        ckpt_path = './reach_and_grasp_weights/reach_shift2/trpo_gail.transition_limitation_2500.SawyerLift.g_step_1.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0/'
        ckpt = tf.compat.v1.train.get_checkpoint_state(ckpt_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
        # Create the playback environment
    
        _, _, last_ob, last_jpos, obs_array, jpos_array = runner_1_traj(play_env,
                pi_reach,
                None,
                timesteps_per_batch=2500,
                number_trajs=1,
                stochastic_policy=args.stochastic_policy,
                save=False
                )

    #obs_array = None
    #jpos_array = None

    print(last_jpos)
    print(last_ob)

    if task =='train':
        play_env.close()

        #init_op2 = tf.compat.v1.global_variables_initializer()
        
        #sess2 = tf.compat.v1.Session(config=tf.ConfigProto())
        #with tf.compat.v1.Session(config=tf.ConfigProto()) as sess2:
        #sess2.run(init_op)
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        #with tf.compat.v1.variable_scope("vf_aktr"):
        vf = NeuralNetValueFunction(ob_dim, ac_dim)
        #with tf.compat.v1.variable_scope("pi_aktr"):
        policy = GaussianMlpPolicy(ob_dim, ac_dim)

        #sess2.run(init_op2)
        
        old_acktr_learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=1500,
            desired_kl=0.001,
            num_timesteps=args.num_timesteps,
            save_per_iter=args.save_per_iter,
            ckpt_dir=args.checkpoint_dir,
            traj_limitation=args.traj_limitation,
            last_ob=obs_array,
            last_jpos=jpos_array,
            animate=True)

        env.close()


    elif task =='evaluate':
       # with tf.compat.v1.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        
        #with tf.compat.v1.variable_scope("vf_aktr"):
        eval_vf = NeuralNetValueFunction(ob_dim, ac_dim)
        #with tf.compat.v1.variable_scope("pi_aktr"):
        eval_policy = GaussianMlpPolicy(ob_dim, ac_dim)

        saver_2 = tf.compat.v1.train.Saver(max_to_keep=5)
        
        #sess2 = tf.compat.v1.Session(config=tf.ConfigProto())
        #with tf.compat.v1.Session(config=tf.ConfigProto()) as sess3:
        with tf.compat.v1.Session() as sess3:
            sess3.run(init_op)
                    
            ckpt_path_2 = './checkpoint/grasp_acktr_rl.transition_limitation_1500.SawyerLift' 
            ckpt_2 = tf.compat.v1.train.get_checkpoint_state(ckpt_path_2)
            saver_2.restore(sess3,ckpt_2.model_checkpoint_path)

            tt = 0

            cum_rew = 0

            ob = last_ob
            prev_ob = np.float32(np.zeros(ob.shape)) # check if indeed starts at all zeros

            obfilter = ZFilter(play_env.observation_space.shape)

            statsu = np.load("./checkpoint/grasp_acktr_rl.transition_limitation_1500.SawyerLift/filter_stats_22953000.npz")

            print("load n: ", statsu["n"])
            print("load M: ", statsu["M"])
            print("load S: ", statsu["S"])

            obfilter.rs._n = statsu["n"]
            obfilter.rs._M = statsu["M"]
            obfilter.rs._S = statsu["S"]

            print("obf n: ", obfilter.rs._n)
            print("obf M: ", obfilter.rs._M)
            print("obf S: ", obfilter.rs._S)
            
            
            ob = obfilter(ob)

            while True:
                s = np.concatenate([ob,prev_ob], -1) 
                ac, _, _ = eval_policy.act(s)
                
                prev_ob = np.copy(ob)
                
                scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
                scaled_ac = np.clip(scaled_ac, play_env.action_space.low, play_env.action_space.high)

                ob, rew, new, _ = play_env.step(scaled_ac)

                ob = obfilter(ob)

                cum_rew += rew

                play_env.render() # check the running in for the first part
                #logger.log("rendering for reach policy")

                if new or tt >= args.traj_limitation:
                    break
                tt += 1
                
            print("Cumulative reward over session: " + str(cum_rew))
        
        play_env.close()


    
    env.close()



def runner_1_traj(env, pi_reach, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

        
        #U.initialize()
        # Prepare for rollouts
        # ----------------------------------------
        #U.load_variables(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    jpos_list = []

    sims_list = [] # For simulations

    traj, last_ob, last_jpos = traj_1_generator(pi_reach, env, timesteps_per_batch, stochastic=stochastic_policy)
    obs, acs, ep_len, ep_ret, jpos = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret'], traj['jpos']
    sims = traj["sims"]# for simulations
    obs_list.append(obs)
    sims_list.append(sims)
    acs_list.append(acs)
    len_list.append(ep_len)
    ret_list.append(ep_ret)
    jpos_list.append(jpos)

    # For env sim playback
    #ii = 0
    #for state_sim in sims:
    #    play_env.sim.set_state_from_flattened(state_sim)
    #    print("Action to see if any go out of range:", acs[ii]) #clip actions
    #    ii += 1
    #    play_env.sim.forward()
    #    play_env.render()
        #time.sleep(0.05)
        #print("state")

    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)

    # Get the last joint positions to load for next part
    #last_jpos = env._get_observation() # just making sure its being loaded correctly
    return avg_len, avg_ret, last_ob, last_jpos, obs, jpos


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi_reach, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()

    #env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]) # to match collected trajectories
    
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    j_pos_a = []

    # Create a sim storage for simulating such trajectory
    sims = []

    while True:
        ac, vpred = pi_reach.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        j_pos_a.append(env._joint_positions) #joint position

        # For simulation playback
        sims.append( env.sim.get_state().flatten() ) # Only works with robosuite environment

        ob, rew, new, _ = env.step(ac)
        #print(ob)
        rews.append(rew)

        env.render() # check the running in for the first part
        #logger.log("rendering for reach policy")

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    last_jpos = env._joint_positions

    obs = np.array(obs)
    sims = np.array(sims) # for simulations
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    j_pos_a = np.array(j_pos_a)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, "sims": sims, "jpos": j_pos_a}
    return traj, ob, last_jpos


if __name__ == '__main__':
    args = argsparser()
    main(args)
