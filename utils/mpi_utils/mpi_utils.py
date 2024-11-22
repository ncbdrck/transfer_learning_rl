from mpi4py import MPI
import numpy as np
import torch


# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync

    """
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode='params')


def sync_grads(network):
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    # todo: get the average grads
    global_grads /= comm.Get_size()
    _set_flat_params_or_grads(network, global_grads, mode='grads')


# get the flat grads or params
def _get_flat_params_or_grads(network, mode='params'):
    """
    include two kinds: grads and params

    """
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    include two kinds: grads and params

    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()


def sync_networks_tl(network, debug=False):
    """
    Synchronize network parameters across all MPI processes.
    This is used for transfer learning where only the trainable layers are synced.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if debug:
        print(f"Rank {rank}: Starting sync_networks_tl")
    for name, param in network.named_parameters():
        if debug:
            print(f"Rank {rank}: Syncing parameter {name}")

        # Get the data of the parameter
        param_data_cpu = param.data.cpu().numpy()

        # Prepare to send the shape of the data array
        shape = np.array(param_data_cpu.shape, dtype=np.int32)  # Ensure you're using np.int64 or np.int32 as needed
        comm.Bcast(shape, root=0)
        # Prepare buffer to receive the data of correct shape
        if not param_data_cpu.shape == tuple(shape):  # Ensure shapes match
            raise ValueError("Shape mismatch in MPI broadcasting")
        # buffer = np.empty(shape, dtype=param_data.dtype)  # Prepare buffer to receive data of the right shape
        comm.Bcast(param_data_cpu, root=0)
        param.data = torch.from_numpy(param_data_cpu).to(param.device)


def sync_grads_tl(network):
    """
    Synchronize gradients across all MPI processes.
    Similar to sync_grads but only for trainable parameters.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    for name, param in network.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_data = param.grad.data.cpu().numpy()
            summed_grad = np.zeros_like(grad_data)
            comm.Allreduce(grad_data, summed_grad, op=MPI.SUM)
            param.grad.data = torch.from_numpy(summed_grad / size).to(param.device)
