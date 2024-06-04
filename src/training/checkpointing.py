import orbax.checkpoint as ocp
from flax.training import orbax_utils


class Checkpointer:
    def __init__(self, path):
        options = ocp.CheckpointManagerOptions(
            step_prefix='ckpt',
        )
        self.manager = ocp.CheckpointManager(
            path,
            ocp.PyTreeCheckpointer(),
            options=options,
        )
            
            
    def save(self, k, params, opt_state):
        checkpoint = {
            'params': params,
            'opt_state': opt_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.manager.save(
            k, 
            checkpoint,
            save_kwargs={'save_args': save_args}
        )
            
    
    def close(self):
        self.manager.close()


# orbax-checkpoint version 0.5.10 an so on would use something like:
# I am living this here until flax updates its documentation on checkpointing

# class Checkpointer:
#     def __init__(self, path):
#         # Sharding
#         self.sharding = jax.sharding.NamedSharding(
#             jax.sharding.Mesh(jax.devices(), ('model',)),
#             jax.sharding.PartitionSpec(
#                 'model',
#             ),
#         )
#         self.create_sharded_array = lambda x: jax.device_put(x, self.sharding)
        
#         # Checkpoint manager
#         options = ocp.CheckpointManagerOptions(step_prefix='epoch',)
#         self.manager = ocp.CheckpointManager(
#             path,
#             item_names=(
#                 'params', 
#                 # 'opt_state',
#             ),
#             options=options,
#         )
        
        
#     def save_epoch(self, epoch, params, opt_state):
#         # Transform params and opt_state into sharded arrays
#         params_pytree = jax.tree_util.tree_map(self.create_sharded_array, params)
#         # opt_state_pytree = jax.tree_util.tree_map(self.create_sharded_array, opt_state)
        
#         # Checkpoint
#         self.manager.save(
#             epoch, 
#             args=ocp.args.Composite(
#                 params=ocp.args.StandardSave(params_pytree),
#                 # opt_state=ocp.args.StandardSave(opt_state_pytree),
#             ),
#         )  
        
    
#     def close(self):
#         self.manager.close()