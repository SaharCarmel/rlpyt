
from rlpyt.utils.seed import set_envs_seeds
from collections import namedtuple
from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.utils.logging import logger
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.collectors import SerialEvalCollector
<< << << < HEAD
== == == =
>>>>>> > upstream/master

EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class SerialSampler(BaseSampler):
    """The simplest sampler; no parallelism, everything occurs in same, master
    Python process.  This can be easier for debugging (e.g. can use
    ``breakpoint()`` in master process) and might be fast enough for
    experiment purposes.  Should be used with collectors which generate the
    agent's actions internally, i.e. CPU-based collectors but not GPU-based
    ones.
    """

    def __init__(self, *args, CollectorCls=CpuResetCollector,
                 eval_CollectorCls=SerialEvalCollector, **kwargs):
        super().__init__(*args, CollectorCls=CollectorCls,
                         eval_CollectorCls=eval_CollectorCls, **kwargs)

    def initialize(
            self,
            agent,
            affinity=None,
            seed=None,
            bootstrap_value=False,
            traj_info_kwargs=None,
            rank=0,
            world_size=1,
    ):
        """Store the input arguments.  Instantiate the specified number of environment
        instances (``batch_B``).  Initialize the agent, and pre-allocate a memory buffer
        to hold the samples collected in each batch.  Applies ``traj_info_kwargs`` settings
        to the `TrajInfoCls` by direct class attribute assignment.  Instantiates the Collector
        and, if applicable, the evaluation Collector.

        Returns a structure of inidividual examples for data fields such as `observation`,
        `action`, etc, which can be used to allocate a replay buffer.
        """
        B = self.batch_spec.B
        envList = self.coach.generateInitialVector()
        envs = [self.EnvCls(seed, level) for level in envList]
        # envs = [self.EnvCls(seed) for _ in range(B)]
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        spaces = EnvSpaces(
            observation=envs[0].observation_space,
            action=envs[0].action_space,
        )
        agent.initialize(spaces, share_memory=False,
                         global_B=global_B, env_ranks=env_ranks)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, envs[0],
                                                                 self.batch_spec, bootstrap_value, agent_shared=False,
                                                                 env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=samples_np,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=agent,
            global_B=global_B,
            env_ranks=env_ranks,  # Might get applied redundantly to agent.
        )
        if self.eval_n_envs > 0:  # May do evaluation.
            eval_envs = [self.EnvCls(0, self.eval_env)
                         for _ in range(self.eval_n_envs)]
            set_envs_seeds(eval_envs, seed)
            eval_CollectorCls = self.eval_CollectorCls or SerialEvalCollector
            self.eval_collector = eval_CollectorCls(
                envs=eval_envs,
                agent=agent,
                TrajInfoCls=self.TrajInfoCls,
                max_T=self.eval_max_steps // self.eval_n_envs,
                max_trajectories=self.eval_max_trajectories,
            )

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        collector.start_agent()

        self.agent = agent
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        self.seed = seed
        logger.log("Serial Sampler initialized.")
        return examples

    def obtain_samples(self, itr):
        """Call the collector to execute a batch of agent-environment interactions.
        Return data in torch tensors, and a list of trajectory-info objects from
        episodes which ended.
        """
        # self.samples_np[:] = 0  # Unnecessary and may take time.
        agent_inputs, traj_infos, completed_infos = self.collector.collect_batch(
            self.agent_inputs, self.traj_infos, itr)
        self.collector.reset_if_needed(agent_inputs)
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        return self.samples_pyt, completed_infos

    def evaluate_agent(self, itr):
        """Call the evaluation collector to execute agent-environment interactions."""
        return self.eval_collector.collect_evaluation(itr)

    def updateEnvs(self,
                   reward,
                   itr,
                   affinity=None,
                   bootstrap_value=True,
                   traj_info_kwargs=None,
                   rank=0,
                   world_size=1,):
        B = self.batch_spec.B
        envList = self.coach.generateVector(reward, itr)
        envs = [self.EnvCls(self.seed, level) for level in envList]
        # envs = [self.EnvCls(seed) for _ in range(B)]
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        spaces = EnvSpaces(
            observation=envs[0].observation_space,
            action=envs[0].action_space,
        )
        samples_pyt, samples_np, examples = build_samples_buffer(self.agent, envs[0],
                                                                 self.batch_spec, bootstrap_value, agent_shared=False,
                                                                 env_shared=False, subprocess=False)
        # if traj_info_kwargs:
        #     for k, v in traj_info_kwargs.items():
        #         setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=samples_np,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=self.agent,
            global_B=global_B,
            env_ranks=env_ranks,  # Might get applied redundantly to agent.
        )

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        # collector.start_agent()

        # self.agent = agent
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
