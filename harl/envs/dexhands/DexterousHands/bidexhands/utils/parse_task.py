# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# Allegro hand

# Meta

from harl.envs.dexhands.DexterousHands.bidexhands.tasks.hand_base.multi_vec_task import (
    MultiVecTaskPython,
)

from harl.envs.dexhands.DexterousHands.bidexhands.utils.config import (
    warn_task_name,
)



def parse_task(args, cfg, sim_params, agent_index):
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    try:
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless,
            agent_index=agent_index,
            is_multi_agent=True,
        )
    except NameError as e:
        print(e)
        warn_task_name()
    env = MultiVecTaskPython(task, rl_device)
    return env
