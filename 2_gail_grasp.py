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
    parser.add_argument('--traj_limitation', type=int, default=2100) # change from -1 to 3000
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
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=1.36e10) # changed to 5e6
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo + "_gail."
    if args.pretrained:
        task_name += "with_pretrained."
    if args.traj_limitation != np.inf:
        task_name += "transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id.split("-")[0]
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
        ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
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
            reward_shaping=True) # Switch from gym to robosuite, also add reward shaping to see reach goal

    env = GymWrapper(env) # wrap in the gym environment

    # Expert Path
    expert_path = '/home/mastercljohnson/Robotics/GAIL_Part/mod_surreal/robosuite/models/assets/demonstrations/120_for_reach/combined/combined_0.npz' # path for 100 trajectories
    #parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    
    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy_sawyer.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    
    #env.seed(args.seed) # Sawyer does not have seed 

    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)

    logger.log("log_directories: ",args.log_dir)
    logger.log("environment action space range: ", env.action_space) #logging the action space
    
    #------- Run policy for reaching ---------#
    # Weights are loaded from reach model grasp_strange


    # Create the playback environment
    play_env = robosuite.make(args.env_id,
            ignore_done=True,
            use_camera_obs=False,
            has_renderer=True,
            control_freq=100,
            gripper_visualization=True)

    #play_env.viewer.set_camera(camera_id=2) # Switch views for eval

    _, _, last_jpos = runner_1_traj(env,
            play_env,
            policy_fn,
            args.load_model_path,
            timesteps_per_batch=3500, # Change time step per batch to be more reasonable
            number_trajs=1, # change from 10 to 1 for evaluation
            stochastic_policy=args.stochastic_policy,
            save=args.save_sample
            )

    

    # ------ Train policy for grasping after reaching the final state of run -------#

    # Set the joint positions in the environment to the expected
    env.set_robot_joint_positions(last_jpos)

    dataset = Mujoco_Dset(expert_path=expert_path, traj_limitation=args.traj_limitation)

    # Check dimensions of the dataset
    #print("dimension of inputs", dataset.dset.inputs.shape) # dims seem correct
    #print("dimension of inputs", dataset.dset.labels.shape) # dims seem correct

    reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
    train(env,
          args.seed,
          policy_fn,
          reward_giver,
          dataset,
          args.algo,
          args.g_step,
          args.d_step,
          args.policy_entcoeff,
          args.num_timesteps,
          args.save_per_iter,
          args.checkpoint_dir,
          args.log_dir,
          args.pretrained,
          args.BC_max_iter,
          task_name
          )
    
    play_env.close()
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter)

    # These are initialized to the same thing always so good
    #logger.log("all positions: \n", env.reset() ) # print the object positions
    #logger.log("all positions: \n", env.reset() ) # print the object positions to see if same
    
    if algo == 'trpo':
        from baselines.gail import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        #env.seed(workerseed) # removed since SawyerLift doesnt have seed

        # Adjustin trpo stuff
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=15000, # changed b=timesteps per batch for scaled env from 10000
                       max_kl=0.001, cg_iters=50, cg_damping=0.1, # maxkl was 0.01, cg iters was 10, cg_dampening from 0.1 
                       gamma=0.995, lam=0.97, # originally 0.97
                       vf_iters=5, vf_stepsize=1e-3,
                       task_name=task_name)
    else:
        raise NotImplementedError

def runner_1_traj(env, play_env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)

    # Hack for loading policies using tensorflow
    init_op = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(max_to_keep=5)
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Load Checkpoint
        ckpt = tf.compat.v1.train.get_checkpoint_state('./grasp_strange/trpo_gail.transition_limitation_2100.SawyerLift.g_step_1.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0/')
        saver.restore(sess, ckpt.model_checkpoint_path)
    
        #U.initialize()
        # Prepare for rollouts
        # ----------------------------------------
        #U.load_variables(load_model_path)

        obs_list = []
        acs_list = []
        len_list = []
        ret_list = []

        sims_list = [] # For simulations

        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        sims = traj["sims"]# for simulations
        obs_list.append(obs)
        sims_list.append(sims)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)

        # For env sim playback
        ii = 0
        for state_sim in sims:
            play_env.sim.set_state_from_flattened(state_sim)
            print("Action to see if any go out of range:", acs[ii]) #clip actions
            ii += 1
            play_env.sim.forward()
            play_env.render()
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
    last_jpos = env._joint_positions # just making sure its being loaded correctly
    return avg_len, avg_ret, last_jpos


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

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

    # Create a sim storage for simulating such trajectory
    sims = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        # For simulation playback
        sims.append( env.sim.get_state().flatten() ) # Only works with robosuite environment

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    sims = np.array(sims) # for simulations
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, "sims": sims,}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
